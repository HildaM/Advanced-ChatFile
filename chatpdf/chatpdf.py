'''
class ChatPDF
'''

import torch
from loguru import logger
from splitter.LangchainSplitter import LangchainSplitter
from vectordb.chroma import FaissDB
from langchain_community.llms import Ollama
from typing import List, Union
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain.globals import set_debug
set_debug(True)


PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
已知内容:
{context_str}

问题:
{query_str}
""")


class ChatPDF:
    def __init__(
        self,
        # similarity_model: SimilarityABC = None,
        generate_model_type: str = "auto",
        generate_model_name_or_path: str = "01-ai/Yi-6B-Chat",
        lora_model_name_or_path: str = None,
        corpus_files: Union[str, List[str]] = None,
        save_corpus_emb_dir: str = "./corpus_embs/",
        device: str = None,
        # int8: bool = False,
        # int4: bool = False,
        chunk_size: int = 250,
        chunk_overlap: int = 0,
        rerank_model_name_or_path: str = None,
        enable_history: bool = False,
        num_expand_context_chunk: int = 2,
        similarity_top_k: int = 10,
        rerank_top_k: int = 3,
    ):
        # 判断设备
        if torch.cuda.is_available():
            default_device = torch.device(0)
        elif torch.backends.mps.is_available():
            default_device = 'mps'
        self.device = device or default_device

        # chunk分割参数验证
        if num_expand_context_chunk > 0 and chunk_overlap > 0:
            logger.warning(f" 'num_expand_context_chunk' and 'chunk_overlap' cannot both be greater than zero. "
                           f" 'chunk_overlap' has been set to zero by default.")
            chunk_overlap = 0
        
        # 文本分割
        # self.text_splitter = SentenceSplitter(chunk_size, chunk_overlap)
        self.text_splitter = LangchainSplitter(chunk_size, chunk_overlap)

        # 向量数据库模型
        self.vectorDB = FaissDB(self.text_splitter, similarity_top_k)
        if corpus_files:
            self.vectorDB.init_corpus(corpus_files)

        self.num_expand_context_chunk = num_expand_context_chunk
        self.rerank_top_k = rerank_top_k

        # 历史记录
        self.history = []
        self.enable_history = enable_history

        # llm设置
        # See: https://python.langchain.com/docs/integrations/llms/ollama
        self.model = Ollama(model="mistral:latest")

        # 输出格式化
        self.output_parser = StrOutputParser()

        self.compression_retriever = None
        self.similarity_top_k = similarity_top_k


    # 获取关联信息的相关度，用于评判召回质量
    def get_reranker_score(self, query: str, reference_results: List[str]) -> List:
        if self.compression_retriever is None:
            # 对reference_results创建临时向量数据库，进行评估
            # db = FaissDB(self.text_splitter, self.similarity_top_k)
            # reference = self.text_splitter.split_text(reference_results)
            # db.add_corpus(reference)

            loader = TextLoader(file_path=file_path)
            document = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(document)
            db = FAISS.from_documents(docs, self.vectorDB.embedding_function)

            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=FlashrankRerank(), base_retriever=db.as_retriever()
            )

        compressed_docs = self.compression_retriever.get_relevant_documents(query)
        return [doc.metadata["id"] for doc in compressed_docs]
    

    # 从向量数据库中获取与query相关联的资料
    def get_reference_results(self, query: str):
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
        vec_contents = self.vectorDB.similarity_search(query)
        for doc in vec_contents:
            reference_results.append(doc.page_content)
        
        # rerank: 对获取的资料进行评估，返回最符合query的topK个
        if reference_results:
            rerank_scores = self.get_reranker_score(query, reference_results)
            logger.debug(f"rerank_scores: {rerank_scores}")

            # 获取topK个符合条件的chunks
            # 将reference_results, rerank_scores通过zip()进行组合，然后按照rerank进行降序排序，输出前k个数据
            reference_results = [reference for reference, score in sorted(
                    zip(reference_results, rerank_scores), key=lambda x: x[1], reverse=True)][:self.rerank_top_k]

        return reference_results
    

    # 一次询问
    def chat(
        self,
        query: str,
        max_length: int = 512,
        context_len: int = 2048,
        temperature: float = 0.7,
    ):
        reference_results = []
        if not self.enable_history:
            self.history = []

        # 1. 检索向量数据库获取信息，并组装prompt
        reference_results = self.get_reference_results(query)
        reference_results = self._add_source_numbers(reference_results)
        context_str = '\n'.join(reference_results)
        logger.info("reference_results: " + str(reference_results))
        
        # 2. llm生成回答
        chain = PROMPT_TEMPLATE | self.model | self.output_parser
        response = chain.invoke({"context_str": context_str, "query_str": query})
        return response
            

    """为参考资料增加索引等信息"""
    @staticmethod
    def _add_source_numbers(lst):
        """Add source numbers to a list of strings."""
        return [f'[{idx + 1}]\t "{item}"' for idx, item in enumerate(lst)]
        

file_path = '/Users/rionzhao/AILearning/rag/MyChatPDF/ChatPDF-langchain/.VSCodeCounter/2024-02-22_11-58-55/details.md'

if __name__ == "__main__":
    chatpdf = ChatPDF(
        corpus_files=file_path
    )

    print(chatpdf.chat("What is the 'Details' mainly about?"))