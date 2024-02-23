'''
Package file: 文件处理工具类
'''
import os
import os
from loguru import logger
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)

# 不同文件的解析器
__LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".json": (TextLoader, {"encoding": "utf8"}),
    ".md": (UnstructuredMarkdownLoader, {}),
}

def __get_loader(path: str, type: str):
    if type in __LOADER_MAPPING:
        loader, args = __LOADER_MAPPING[type]
        return loader(path, **args)
    else:
        logger.warning("Unsupported file type: " + type)
    


def __load_file(path: str):
    file_type = os.path.splitext(path)[1]
    loader = __get_loader(path, file_type)
    if loader is None:
        return None
    return loader.load()


# 获取文件夹下的全部文件名(遍历给定目录下的所有子目录和文件)
def __get_all_paths(directory):
    file_paths = []  # 用于存储文件路径的列表
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(os.path.abspath(filepath))
    
    return file_paths


# 加载指定文件夹下的全部文件
def load_from_folder(root_path: str):
    documents = []
    for file_path in __get_all_paths(root_path):
        # 根据文件类型选择合适的解析方式，如果文件类型不支持则抛出warning提示
        docs = __load_file(file_path)
        if docs is None:
            continue
        for doc in docs:
            documents.append(doc)
    
    return documents


# 加载单个文件
def load_single_file(path: str):
    return __load_file(path)
    