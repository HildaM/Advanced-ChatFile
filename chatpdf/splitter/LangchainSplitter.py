from typing import List
from langchain.text_splitter import CharacterTextSplitter

class LangchainSplitter:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if self._isHasChinese(text):
            return self._splitChineseText(text)
        else:
            return self._splitEnglishText(text)
        
    def _isHasChinese(self, text: str) -> bool:
        # check if contains chinese characters
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return True
        else:
            return False
    
    def _splitChineseText(self, text: str) -> List[str]:
        sentence_endings = {'\n', '。', '！', '？', '；', '…'}  # 句末标点符号
        splitter = CharacterTextSplitter(
            separator=sentence_endings,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_text(text)
    
    def _splitEnglishText(self, text: str) -> List[str]:
        sentence_endings = {'\n', '\n\n', '.', '!', '?', ';', '…'}  # 句末标点符号
        splitter = CharacterTextSplitter(
            separator=sentence_endings,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_text(text)

