from typing import Union, List
from dotenv import load_dotenv
import os
from .config_keys import *


"""单例模式，确保全局只有一个Config"""
class Singleton(type):
    _instance = {}

    # 重写了 __call__ 方法，该方法在类实例化时被调用。
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class BaseSingleton(metaclass=Singleton):
    pass

"""基类"""
class BaseObject(BaseSingleton):
    @classmethod
    def class_name(cls):
        return cls.__name__
    

"""Config 配置类"""
class Config(BaseObject):
    def __init__(
        self,
        # Ollama LLM Settings
        llm_model_name: str = "mistral:latest",
        model_max_input_size: int = 5,
        waiting_time: float = 2,
        enable_history: bool = True,
        # Embedding Model Settings
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        files_path: Union[str, List[str]] = None,
        refresh_vectordb: bool = True,
        device: str = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        # Reranker Settings
        rerank_model_name: str = "BAAI/bge-reranker-large",
        similarity_top_k: int = 10,
        rerank_top_k: int = 3,
        # MongoDB Settings
        memory_connection_string: str = None,
        memory_database_name: str = None,
        memory_collection_name: str = None,
        session_id: str = None,
        mongo_username: str = None,
        mongo_password: str = None,
        mongo_cluster: str = None,
        memory_window_size: int = 5
    ):
        super().__init__()
        # 加载配置文件
        load_dotenv('.env')
        self.llm_model_name = llm_model_name if llm_model_name is not None else os.getenv(LLM_MODEL_NAME)