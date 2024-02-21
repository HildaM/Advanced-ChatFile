from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores.chroma import Chroma
from typing import Union, Dict, List


class ChromaDB:
    def __init__(self, docs: str) -> None:
        # TODO embedding作为入参输入
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma.from_texts(docs, embedding_function)

    # 增加语料库
    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
        self.db.add_texts(corpus)