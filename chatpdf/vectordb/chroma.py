from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores.chroma import Chroma
from typing import Union, Dict, List, Optional, Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# 常量
DEFAULT_K = 4


class ChromaDB:
    def __init__(self, splitter, similarity_top_k: int = 20) -> None:
        self.similarity_top_k = similarity_top_k

        # embedding function
        # create the open-source embedding function
        self.embedding_function = self.get_embedding_function()
        self.splitter = splitter
        self.db = None


    # TODO 将model选择与device进行定制化
    def get_embedding_function(self) -> HuggingFaceEmbeddings:
        # Embedding Model
        modelPath = "BAAI/bge-base-en-v1.5"

        # Create a dictionary with model configuration options, specifying to use the CPU for computations
        # model_kwargs = {'device':'cpu'}
        model_kwargs = {'device':'mps'}

        # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
        encode_kwargs = {'normalize_embeddings': True}

        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,     # Provide the pre-trained model's path
            model_kwargs=model_kwargs, # Pass the model configuration options
            encode_kwargs=encode_kwargs # Pass the encoding options
        )
        return embeddings


    def add_corpus(self, files: Union[str, List[str]]):
        '''加载文件'''
        if isinstance(files, str):
            files = [files]
        
        self.db = Chroma.from_documents(files, self.embedding_function)


    def get_retriever(self):
        return self.db.as_retriever(search_kwargs={"k": self.similarity_top_k})
    
    
    def similarity_search(self, query: str) -> List[Document]:
        return self.db.similarity_search(query, self.similarity_top_k)
