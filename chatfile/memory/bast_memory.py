from common.entity import Message
import os
from common.default_configs import PROJECT_ROOT
import json
from common.config import Config
from loguru import logger

LOG_PATH = PROJECT_ROOT + "/log"

"""
BaseMemory: 将历史聊天记录存储在本地文件中
"""
class BaseMemory:
    def __init__(self, config: Config = None):
        self._config = config if config is not None else Config()
        self._latest_history_nums = self._config.latest_history_nums
        self._conversation_id = 0
        self._messages = []
        self._filepath = ""


    """清空历史"""
    def clear(self, conversation_id: str):
        self._messages.clear()


    """加载历史
        如果本地存在日志文件，则加载历史数据
        如果不存在，则先创建文件
    """
    def load_history(self, conversation_id: str):
        self._conversation_id = conversation_id
        filename = LOG_PATH + "/" + f"{conversation_id}.log"

        # 1. 检查目录是否存在
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # 2. 检查文件是否存在：存在则加载数据；不存在则创建空json文件
        self._filepath = filename
        if os.path.isfile(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
                for item in data:
                    self._messages.append(Message(item["Question"], item["Answer"]))
        else:
            # 创建空json文件
            with open(filename, 'w') as file:
                file.write('[]')

        logger.info("load history: " + str(self._messages))
    

    """增加历史信息"""
    def add_history(self, message: Message):
        self._messages.append(Message(question=message.question, answer=message.answer))

    
    """更新本地历史数据存档"""
    def save_history(self):
        data = [{"Question": msg.question, "Answer": msg.answer} for msg in self._messages]
        with open(self._filepath, 'w', encoding='utf-8') as file:   # 指定文件编码为UTF-8
            json.dump(data, file, indent=3, ensure_ascii=False)     # ensure_ascii=False 保证中文不会出现乱码

    
    """获取最后k个历史记录"""
    def get_latest(self) -> str:
        messages =  self._messages[-self._latest_history_nums:]
        return '\n'.join(str(msg) for msg in messages)