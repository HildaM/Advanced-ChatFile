from langchain_community.vectorstores.faiss import FAISS
from typing import Union, Dict, List, Optional, Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from file import extract
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader


# 常量
DEFAULT_K = 4


class FaissDB:
    def __init__(self, splitter, similarity_top_k: int = 20) -> None:
        self.similarity_top_k = similarity_top_k

        # embedding function
        self.embedding_function = self.get_embedding_function()
        self.splitter = splitter
        self.db = None

    # TODO 将model选择与device进行定制化
    def get_embedding_function(self) -> HuggingFaceEmbeddings:
        # Embedding Model
        modelPath = "BAAI/bge-base-en-v1.5"

        # Create a dictionary with model configuration options, specifying to use the CPU for computations
        # model_kwargs = {'device':'cpu'}
        model_kwargs = {"device": "mps"}

        # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
        encode_kwargs = {"normalize_embeddings": True}

        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,  # Provide the pre-trained model's path
            model_kwargs=model_kwargs,  # Pass the model configuration options
            encode_kwargs=encode_kwargs,  # Pass the encoding options
        )
        return embeddings

    def init_files(self, file_path: str):
        """加载文件"""
        # TODO 此处需要补充“文件类型适配层”，方便适配不同文件的加载
        loader = TextLoader(file_path)
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        documents = splitter.split_documents(documents)
        self.db = FAISS.from_documents(documents, self.embedding_function)

    def add_corpus(self, data: Union[str, List[str]]):
        """Load document files."""
        # if isinstance(files, str):
        #     files = [files]

        self.db.add_texts(data)

        # 解析文档
        # for doc_file in files:
        #     chunks = self._chunk_files(doc_file)
        #     self.db.add_texts(chunks)

    def _chunk_files(self, doc_file: str) -> str:
        if doc_file.endswith(".pdf"):
            corpus = extract.from_pdf(doc_file)
        elif doc_file.endswith(".docx"):
            corpus = extract.from_docx(doc_file)
        elif doc_file.endswith(".md"):
            corpus = extract.from_markdown(doc_file)
        else:
            corpus = extract.from_txt(doc_file)

        full_text = "\n".join(corpus)
        chunks = self.splitter.split_text(full_text)

        return chunks

    def get_retriever(self):
        return self.db.as_retriever(search_kwargs={"k": self.similarity_top_k})

    def similarity_search(self, query: str) -> List[Document]:
        return self.db.similarity_search(query, self.similarity_top_k)
