from memory.bast_memory import BaseMemory
from common.config import Config
from langchain.memory import MongoDBChatMessageHistory

class MongoMemory(BaseMemory):
    def __init__(self, config: Config = None, **kwargs):
        config = config if config is not None else Config()
        super(MongoMemory, self).__init__(
            config=config,
            chat_history_class=MongoDBChatMessageHistory,
            chat_history_kwargs={
                "connection_string": config.memory_connection_string,
                "session_id": config.session_id,
                "database_name": config.memory_database_name,
                "collection_name": config.memory_collection_name
            }
        )