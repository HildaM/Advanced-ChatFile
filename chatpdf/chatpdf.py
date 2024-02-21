'''
class ChatPDF
'''

import torch
from loguru import logger
from langchain.text_splitter import LangchainSplitter
from vectordb.chroma import ChromaDB

class ChatPDF:
    def __init__(
        self,
        # similarity_model: SimilarityABC = None,
        generate_model_type: str = "auto",
        generate_model_name_or_path: str = "01-ai/Yi-6B-Chat",
        lora_model_name_or_path: str = None,
        # corpus_files: Union[str, List[str]] = None,
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
        # TODO 此处不能传递参数，因为docs在初始化的时候根本不会传入进去
        self.sim_model = ChromaDB

