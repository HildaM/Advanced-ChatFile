from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory
from typing import Optional
from common.entity import Message


class BaseMemory:
    # __slots__ = ["_base_memory", "_memory"] 

    def __init__(
            self,
            chat_history_class=ChatMessageHistory,
            memory_class=ConversationBufferWindowMemory,
            chat_history_kwargs: Optional[dict] = None,
            # **kwargs
    ):
        """
        :param chat_history_class: LangChain's chat history class
        :param memory_class: LangChain's memory class
        :param kwargs: Memory class kwargs
        """
        # self._params = kwargs
        self._chat_history_kwargs = chat_history_kwargs or {}
        self._base_memory_class = chat_history_class
        self._memory = memory_class(**self.params)
        self._user_memory = dict()

    @property
    def params(self):
        # if self._params:
        #     return self._params
        return {
            # TODO: 此处后续需要做配置化处理
            "ai_prefix": "AI",
            "human_prefix": "Human",
            "memory_key": "History",
            "k": 3
        }
    
    # @property
    # def memory(self):
    #     return self._memory
    
    # @property
    # def user_memory(self):
    #     return self._user_memory
    

    """清空历史"""
    def clear(self, conversation_id: str):
        if conversation_id in self._user_memory:
            memory = self._user_memory.pop(conversation_id)
            memory.clear()

    """加载历史"""
    def load_history(self, conversation_id: str) -> str:
        if conversation_id not in self._user_memory:
            # 初始化，并返回空值
            memory = self._base_memory_class(**self._chat_history_kwargs)
            self._memory.chat_memory = memory
            self._user_memory[conversation_id] = memory
            return ""
        
        self._memory.chat_memory = self._user_memory.get(conversation_id)
        return self._memory.load_memory_variables({})["history"]
    
    """增加历史信息"""
    def add_history(self, conversation_id: str, message: Message):
        memory = self._user_memory[conversation_id]
        memory.add_message({"Question": message.human_req, "Answer": message.ai_resp})