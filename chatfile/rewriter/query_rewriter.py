from langchain_core.prompts import ChatPromptTemplate
from llm.llm_adapter import LLMAdapter
from memory.bast_memory import BaseMemory
from langchain_core.output_parsers import StrOutputParser

REWRITER_PROMPT = ChatPromptTemplate.from_template(
"""Given a question and its context and a rewrite that decontextualizes thequestion, edit the rewrite to create a revised version that fully addressescoreferences and omissions in the question without changing the originalmeaning of the question but providing more information. The new rewriteshould not duplicate any previously asked questions in the context, Ifthere is no need to edit the rewrite, return the rewrite as-is.
Previous Chat History:
"{previous}"
Current Query:
"{current}"
Only output the rewritten question! DON'T add any prefix, such as "Rewritten Question: "rewritten question?".
"""
)


class QueryRewriter:

    def __init__(
        self, 
        model_name: str, 
        conversation_id: str,
        memory: BaseMemory,
    ):
        # Rewriter 的 LLM 应该支持自定义，与 ChatFile 本体的 LLM 分开处理
        self._llm = LLMAdapter(model_name).build()
        self._conversation_id = conversation_id
        self._memory = memory
        self._output_parser = StrOutputParser()

    def rewrite(self, query: str) -> str:
        chain = REWRITER_PROMPT | self._llm | self._output_parser
        response = chain.invoke({"previous": self._memory.get_latest(), "current": query})
        return response
