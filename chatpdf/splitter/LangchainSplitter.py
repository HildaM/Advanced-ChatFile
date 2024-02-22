from typing import List
from langchain.text_splitter import CharacterTextSplitter
import re

class LangchainSplitter:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if self._isHasChinese(text):
            return self._splitChineseText(text)
        else:
            return self._split_english_text(text)
        
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
        sentence_endings = {'\n', '\n\n', '.', '!', '?', ';', '…'}  # 句末标点符号x
        splitter = CharacterTextSplitter(
            separator=sentence_endings,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_text(text)
    

    def _split_english_text(self, text: str) -> List[str]:
        # 使用正则表达式按句子分割英文文本
        sentences = re.split(r'(?<=[.!?])\s+', text[0].replace('\n', ' '))
        chunks, current_chunk = [], ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size or not current_chunk:
                current_chunk += (' ' if current_chunk else '') + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:  # Add the last chunk
            chunks.append(current_chunk)

        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)

        return chunks

