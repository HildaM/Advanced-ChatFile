import os
from langchain_openai import OpenAI
from langchain_community.llms import Ollama

class LLMAdapter:
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.args = {}
        self.__load_model_specific_settings()

    def __load_model_specific_settings(self):
        # 根据模型名称构造环境变量前缀
        prefix = self.model_name.upper().split(":")[0] + "_"    # 适配入参携带版本号的情况。如ollama:latest
        # 遍历环境变量，加载与模型相关的配置
        for key in os.environ:
            print("os.environ: " + key)
            if key.startswith(prefix):
                self.args[key.lower()] = os.getenv(key)
        
        # debug
        for arg in self.args:
            print(arg)

    # 加载模型
    def build(self):
        match self.model_name:
            case "openai":
                return OpenAI(openai_api_base=self.args["openai_api_base"], openai_api_key=self.args["openai_api_key"])
            case "ollama":
                return Ollama(model=self.args["ollama_model_name"])
            case _:
                raise ValueError("Unsupport LLM model type!")