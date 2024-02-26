from typing import Union, List
from dotenv import load_dotenv
import os
from .default_configs import *


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
        enable_history: bool = True,
        # Embedding Model Settings
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        embedding_files_path: Union[str, List[str]] = None,
        refresh_vectordb: bool = True,
        device: str = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        # Reranker Settings
        rerank_model_name: str = "BAAI/bge-reranker-large",
        similarity_top_k: int = 10,
        rerank_top_k: int = 3,
        # History Memory Settings
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
        # Ollama LLM Settings
        self.llm_model_name = llm_model_name if llm_model_name is not None else os.getenv(LLM_MODEL_NAME)
        self.enable_history = enable_history if enable_history is not None else os.getenv(ENABLE_HISTROY)
        # Embedding Model Settings
        self.embedding_model_name = embedding_model_name if embedding_model_name is not None else os.getenv(EMBEDDING_MODEL_NAME)
        self.embedding_files_path = embedding_files_path if embedding_files_path is not None else os.getenv(EMBEDDING_FILE_PATH)
        self.refresh_vectordb = refresh_vectordb if refresh_vectordb is not None else os.getenv(REFRESH_VECTORDB)
        self.device = device if device is not None else os.getenv(DEVICE)
        self.chunk_size = chunk_size if chunk_size is not None else os.getenv(CHUNK_SIZE)
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else os.getenv(CHUNK_OVERLAP)
        # Reranker Settings
        self.rerank_model_name = rerank_model_name if rerank_model_name is not None else os.getenv(RERANK_MODEL_NAME)
        self.similarity_top_k = similarity_top_k if similarity_top_k is not None else os.getenv(SIMILARITY_TOP_K)
        self.rerank_top_k = rerank_top_k if rerank_top_k is not None else os.getenv(RERANK_TOP_K)
        # History Memory Settings
        self.memory_database_name = memory_database_name if memory_database_name is not None \
            else os.getenv(MONGO_DATABASE, "langchain_bot")
        self.memory_collection_name = memory_collection_name if memory_collection_name is not None \
            else os.getenv(MONGO_COLLECTION, "chatbot")
        self.memory_connection_string = memory_connection_string if memory_connection_string is not None \
            else os.getenv(MONGO_CONNECTION_STRING,
                           f"mongodb+srv://{self.mongo_username}:{self.mongo_password}@{self.mongo_cluster}.xnkswcg.mongodb.net")
        self.session_id = session_id if session_id is not None else os.getenv(SESSION_ID)
        self.mongo_username = mongo_username if mongo_username is not None else os.getenv(MONGO_USERNAME)
        self.mongo_password = mongo_password if mongo_password is not None else os.getenv(MONGO_PASSWORD)
        self.mongo_cluster = mongo_cluster if mongo_cluster is not None else os.getenv(MONGO_CLUSTER)
        self.memory_window_size = memory_window_size if memory_window_size is not None else os.getenv(MEMORY_WINDOW_SIZE)
        # Other Settings
        self.ai_prefix = os.getenv(AI_PREFIX, "AI")
        self.human_prefix = os.getenv(HUMAN_PREFIX, "Human")
        self.memory_key = os.getenv(MEMORY_KEY, "history")