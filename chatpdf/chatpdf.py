"""
class ChatPDF
"""

import torch
from loguru import logger
from vectordb.chroma import FaissDB
from langchain_community.llms import Ollama
from typing import List, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_core.messages import HumanMessage

from langchain.globals import set_debug
set_debug(True)


PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """基于以下已知信息，简洁和专业的来回答用户的问题,同时提供相关文档依据来佐证自己的观点。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请根据“参考信息”的语言进行选择。

问题:
{query_str}

参考信息:
{context_str}
"""
)

SPERATORS = ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
            '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']


class ChatPDF:
    def __init__(
        self,
        model_name: str = "mistral:latest",
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        files_path: Union[str, List[str]] = None,
        save_corpus_emb_dir: str = "./corpus_embs/",
        device: str = None,
        chunk_size: int = 250,
        chunk_overlap: int = 0,
        rerank_model_name: str = "BAAI/bge-reranker-large",
        enable_history: bool = False,
        num_expand_context_chunk: int = 2,
        similarity_top_k: int = 10,
        rerank_top_k: int = 3,
    ):
        # 判断设备
        if torch.cuda.is_available():
            default_device = torch.device(0)
        elif torch.backends.mps.is_available():
            default_device = "mps"
        self._device = device or default_device

        # chunk分割参数验证
        if num_expand_context_chunk > 0 and chunk_overlap > 0:
            logger.warning(
                f" 'num_expand_context_chunk' and 'chunk_overlap' cannot both be greater than zero. "
                f" 'chunk_overlap' has been set to zero by default."
            )
            chunk_overlap = 0

        # 文本分割
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SPERATORS
        )

        # rerank 模型
        if rerank_model_name:
            self._rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
            self._rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
            self._rerank_model.to(self._device)
            self._rerank_model.eval()
        

        # 向量数据库模型
        self._vectorDB = FaissDB(self._text_splitter, embedding_model_name, self._device, similarity_top_k)
        # 初始化原始文件
        if files_path:
            self._vectorDB.init_files(files_path)

        self._num_expand_context_chunk = num_expand_context_chunk
        self._rerank_top_k = rerank_top_k

        # 历史记录
        self._history = []
        self._enable_history = enable_history

        # llm设置
        # See: https://python.langchain.com/docs/integrations/llms/ollama
        self._model = Ollama(model=model_name)

        # 输出格式化
        self._output_parser = StrOutputParser()

        self._similarity_top_k = similarity_top_k
    

    """
    获取关联信息的相关度，用于评判召回质量
    https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker
    """
    def _get_reranker_score(self, query: str, reference_results: List[str]):
        """Get reranker score."""
        pairs = []
        for reference in reference_results:
            pairs.append([query, reference])

        with torch.no_grad():
            inputs = self._rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs_on_device = {k: v.to(self._rerank_model.device) for k, v in inputs.items()}
            scores = self._rerank_model(**inputs_on_device, return_dict=True).logits.view(-1, ).float()

        return scores


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

        # rerank: 对获取的资料进行评估，返回最符合query的topK个
        if reference_results:
            rerank_scores = self._get_reranker_score(query, reference_results)
            logger.info("reranker scores: " + str(rerank_scores))

            # 获取topK个符合条件的chunks
            # 将reference_results, rerank_scores通过zip()进行组合，然后按照rerank进行降序排序，输出前k个数据
            reference_results = [
                reference
                for score, reference in sorted(
                    zip(rerank_scores, reference_results),
                    reverse=True,
                )
            ][: self._rerank_top_k]  # 对数据进行切片，只获取前K个

        # 对数据进行处理，增加编号
        reference_results = self._add_source_numbers(reference_results)
        return reference_results
    

    """为参考资料增加索引等信息"""
    @staticmethod
    def _add_source_numbers(lst):
        """Add source numbers to a list of strings."""
        return [f'[{idx + 1}]\t"{item}"' for idx, item in enumerate(lst)]


    """一次询问"""
    def predict(
        self,
        query: str,
        max_length: int = 512,
        context_len: int = 2048,
        temperature: float = 0.7,
    ):
        reference_results = []
        if not self._enable_history:
            self._history = []

        # 1. 检索向量数据库获取信息，并组装prompt
        reference_results = self._get_reference_results(query)
        context_str = "\n".join(reference_results)
        logger.info("reference_results: " + str(reference_results))

        # 2. llm生成回答
        chain = PROMPT_TEMPLATE | self._model | self._output_parser
        response = chain.invoke({"context_str": context_str, "query_str": query})
        if self._enable_history:
            self._history.extend([HumanMessage(content=query), response])
        return response, context_str





if __name__ == "__main__":
    file_path = "../test/I_have_a_dream.txt"
    chatpdf = ChatPDF(files_path=file_path, model_name="mistral:latest", rerank_top_k=4)

    resp, ref = chatpdf.predict("what is the dream of Martin Luther King based on the reference data?")
    print(resp, ref)
