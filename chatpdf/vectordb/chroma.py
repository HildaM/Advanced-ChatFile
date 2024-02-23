from langchain_community.vectorstores.chroma import Chroma
from typing import Union, List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from file import file


# 常量
DEFAULT_K = 4


class FaissDB:
    def __init__(
        self, 
        splitter, 
        model_name: str = "BAAI/bge-base-en-v1.5", 
        device: str = "cpu",
        similarity_top_k: int = 20,
        normalize_embeddings: bool = True,
    ) -> None:

        self._similarity_top_k = similarity_top_k

        # embedding function
        self._embedding_function = self.__get_embedding_function(model_name, device, normalize_embeddings)
        self._splitter = splitter
        self._db = None


    def __get_embedding_function(self, model_name, device, normalize_embeddings) -> HuggingFaceEmbeddings:
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": normalize_embeddings}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,  # Provide the pre-trained model's path
            model_kwargs=model_kwargs,  # Pass the model configuration options
            encode_kwargs=encode_kwargs,  # Pass the encoding options
        )
        return embeddings


    def init_files(self, folder_path: str):
        """加载文件"""
        documents = file.load_from_folder(folder_path)
        documents = self._splitter.split_documents(documents)
        self._db = Chroma.from_documents(documents, self._embedding_function)


    # TODO 不清楚是否删除
    def add_corpus(self, data: Union[str, List[str]]):
        """Load document files."""
        # if isinstance(files, str):
        #     files = [files]

        self._db.add_texts(data)

        # 解析文档
        # for doc_file in files:
        #     chunks = self._chunk_files(doc_file)
        #     self._db.add_texts(chunks)

    # TODO 不清楚是否删除
    # def _chunk_files(self, doc_file: str) -> str:
    #     if doc_file.endswith(".pdf"):
    #         corpus = extract.from_pdf(doc_file)
    #     elif doc_file.endswith(".docx"):
    #         corpus = extract.from_docx(doc_file)
    #     elif doc_file.endswith(".md"):
    #         corpus = extract.from_markdown(doc_file)
    #     else:
    #         corpus = extract.from_txt(doc_file)

    #     full_text = "\n".join(corpus)
    #     chunks = self._splitter.split_text(full_text)

    #     return chunks

    def get_retriever(self):
        return self._db.as_retriever(search_kwargs={"k": self._similarity_top_k})

    def similarity_search(self, query: str) -> List[Document]:
        return self._db.similarity_search(query, self._similarity_top_k)
