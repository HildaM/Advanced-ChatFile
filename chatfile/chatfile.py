import torch
from loguru import logger
from vectordb.chroma import ChromaDB
from typing import List, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import random
from memory.bast_memory import BaseMemory
from common.entity import Message
from common.config import Config
from rewriter.query_rewriter import QueryRewriter
from reranker.reranker import Reranker


# 设置 Langchain Debug 模式
from langchain.globals import set_debug
set_debug(True)

# PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
# """
# 你是一个善于分析问题,拥有严谨推理能力的助理,你需要参考"参考信息"和"上一次的对话信息",回答: "{query_str}" ?
# 这个问题需要你先基于你自己的知识做出初步的判断,然后再基于以下已知信息,对比分析并得出答案,同时提供相关依据来佐证自己的观点.
# 如果无法从中得到答案,请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息",不允许在答案中添加编造成分.

# "参考信息": {context_str} . 必须保持对"参考信息"的辩证思考:如果"参考信息"与问题没有关联,请忽视它直接丢弃,同时不要在回答中提起它,也不要让我知道.

# "上一次的对话消息": {chat_history} . 必须保持对"上一次的对话消息"的辩证思考:如果"上一次对话的信息"与问题没有关联,请忽视它直接丢弃,同时不要在回答中提起它,也不要让我知道.

# 请综合'问题'与'参考信息'的语言种类,作为你回答问题的语言,不要回答与 "{query_str}" 无关的内容!

# Let's think step by step, take a deep breath.
# """
# )

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
"""
You are an assistant skilled in analyzing problems and possessing rigorous reasoning abilities. You need to refer to the "reference information" and "previous conversation information" to answer: "{query_str}"? This question requires you to first make a preliminary judgment based on your own knowledge, then compare and analyze based on the following known information to arrive at an answer, while providing relevant evidence to support your viewpoint. If it is impossible to find an answer from the information provided, please say "Unable to answer the question based on the information available" or "Insufficient relevant information provided," and do not include fabricated content in your answer.
"Reference information": {context_str}. You must maintain dialectical thinking regarding the "reference information": if the "reference information" is not related to the question, please ignore it and do not mention it in your answer, nor let me know about it.
"Previous conversation message": {chat_history}. You must maintain dialectical thinking regarding the "previous conversation message": if the information from the "previous conversation" is not related to the question, please ignore it and do not mention it in your answer, nor let me know about it.
Please use the language of the 'question' and 'reference information' combined as the language for your answer, and do not address content unrelated to "{query_str}".
"""
)



SPERATORS = ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
            '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']



class ChatFile:
    def __init__(
        self,
        config: Config = None,
        files_path: Union[str, List[str]] = None,
    ):
        # 加载配置
        self._config = config if config is not None else Config()

        # 判断设备
        if torch.cuda.is_available():
            default_device = torch.device(0)
        elif torch.backends.mps.is_available():
            default_device = "mps"
        self._device = self._config.device or default_device

        # 文本分割
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size, chunk_overlap=self._config.chunk_overlap, separators=SPERATORS
        )

        # rerank 模型    
        self._reranker = Reranker(self._config, self._device)

        # 向量数据库模型
        self._vectorDB = ChromaDB(self._text_splitter, self._config.embedding_model_name, self._device, self._config.similarity_top_k)
        # 初始化原始文件
        self._vectorDB.init_files(files_path, self._config.refresh_vectordb)
        # RAG 设置
        self._similarity_top_k = self._config.similarity_top_k

        # 历史记录
        self._memory = BaseMemory()
        self._enable_history = self._config.enable_history
        if self._enable_history:
            self._conversating_id = self.create_conversation_id()
            self._memory.load_history(self._conversating_id)

        # llm设置
        # See: https://python.langchain.com/docs/integrations/llms/ollama
        # self._model = Ollama(model=model_name)
        self._model = self._config.llm_model
        self._rewriter = QueryRewriter(self._config.llm_name, self._conversating_id, self._memory, self._reranker)

        # 输出格式化
        self._output_parser = StrOutputParser()

    
    @staticmethod
    def create_conversation_id():
        return str(random.randint(100000000, 999999999))


    """
    从向量数据库中获取与query相关联的资料
    """
    def _get_reference_results(self, query: str):
        """
        Get reference results.
            1. Similarity model get similar chunks
            2. Rerank similar chunks
            3. Expand reference context chunk
        :param query:
        :return:
        """
        reference_results = []
        # 1. 搜索向量数据库
        vec_contents = self._vectorDB.similarity_search(query)
        for doc in vec_contents:
            reference_results.append(doc.page_content)
        logger.info("vec_contents: " + str(reference_results))

        # 2. rerank: 对获取的资料进行评估，返回最符合query的topK个
        if reference_results:
            rerank_scores = self._reranker.get_reranker_score(query, reference_results)
            logger.info("reranker scores: " + str(rerank_scores))

            # 获取topK个符合条件的chunks
            # 将reference_results, rerank_scores通过zip()进行组合，然后按照rerank进行降序排序，输出前k个数据
            reference_results = [
                reference
                for score, reference in sorted(
                    zip(rerank_scores, reference_results),
                    reverse=True,
                )
            ][: self._reranker.top_k()]  # 对数据进行切片，只获取前K个

        # 3. 对数据进行处理，增加编号
        reference_results = self._add_source_numbers(reference_results)
        return reference_results
    

    """Add Index for Retrieval Data"""
    @staticmethod
    def _add_source_numbers(lst):
        """Add source numbers to a list of strings."""
        return [f'[{idx + 1}]\t"{item}"' for idx, item in enumerate(lst)]
    

    """Update Single File"""
    def add_single_file(self, path: str):
        self._vectorDB.add_single_file(path)


    """One Query"""
    def predict(
        self,
        query: str,
        max_length: int = 512,
        context_len: int = 2048,
        temperature: float = 0.7,
    ):
        # 1. Query Rewriter
        r_query = self._rewriter.rewrite(query)
        rewrite_query = r_query if r_query is not None else query
        # rewrite_query = query

        reference_results = []
        # 2. Retrieval and Reranking
        reference_results = self._get_reference_results(rewrite_query)
        context_str = "\n".join(reference_results)
        logger.info("reference_results: " + str(reference_results))

        # 3. LLM generator
        chain = PROMPT_TEMPLATE | self._model | self._output_parser
        if self._enable_history:
            response = chain.invoke({"context_str": context_str, "query_str": rewrite_query, "chat_history": self._memory.get_latest()})
            self._memory.add_history(Message(question=rewrite_query, answer=response))
        else:
            response = chain.invoke({"context_str": context_str, "query_str": rewrite_query})
            
        # TODO Test! Save history
        self._memory.save_history()

        return response, context_str