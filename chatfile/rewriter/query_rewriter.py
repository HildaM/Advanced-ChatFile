from langchain_core.prompts import ChatPromptTemplate
from llm.llm_adapter import LLMAdapter
from memory.bast_memory import BaseMemory
from langchain_core.output_parsers import StrOutputParser
from reranker.reranker import Reranker

REWRITER_PROMPT = ChatPromptTemplate.from_template(
"""Given a question and its context and a rewrite that decontextualizes thequestion, edit the rewrite to create a revised version that fully addressescoreferences and omissions in the question without changing the originalmeaning of the question but providing more information. The new rewriteshould not duplicate any previously asked questions in the context, Ifthere is no need to edit the rewrite, return the rewrite as-is.
Previous Chat History:
"{chat_history}"
Current Query:
"{current}"
Only output the rewritten question! DON'T add any prefix. Just answer the final result.
"""
)

# EVAL_PROMPT = ChatPromptTemplate.from_template(
# """Please analyze whether "{rewrite}" is semantically similar to "{query}", using "{query}" as the primary criterion. If the meanings are not similar, return 0; if the meanings are similar, return 1.
# You are restricted to answer only 0 or 1.
# """
# )


class QueryRewriter:

    def __init__(
        self, 
        model_name: str, 
        conversation_id: str,
        memory: BaseMemory,
        reranker: Reranker,
    ):
        # Rewriter 的 LLM 应该支持自定义，与 ChatFile 本体的 LLM 分开处理
        self._llm = LLMAdapter(model_name).build()
        self._conversation_id = conversation_id
        self._memory = memory
        self._output_parser = StrOutputParser()
        self._reranker = reranker

    def rewrite(self, query: str) -> str:
        chain = REWRITER_PROMPT | self._llm | self._output_parser
        chat_history = self._memory.get_latest()
        if chat_history == "" or chat_history is None:
            chat_history = "Empty"
        rewrite = chain.invoke({"chat_history": chat_history, "current": query})

        # Eval rewrite result using LLM
        # eval_chain = EVAL_PROMPT | self._llm | self._output_parser
        # resp = eval_chain.invoke({"rewrite": rewrite, "query": query})
        # print("eval resp: " + str(resp))
        # if resp != "0":
        #     return query
        return rewrite

        
