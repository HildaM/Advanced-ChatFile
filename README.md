# Advanced-ChatFile

The ChatFile project can parse various file types and use a vector database to retrieve questions, which are then answered by an LLM (Large Language Model).

This is project is the pratise of the article [Mastering RAG: How To Architect An Enterprise RAG System](https://www.rungalileo.io/blog/mastering-rag-how-to-architect-an-enterprise-rag-system)

![image](/docs/architecture.png)


## NOTE!

This is project is still programing, it may have some bug and problem!

If you want to study the code or run the project, you just need place file data into 'test' folder for vectorDB store.

And then, run `python main.py`. You will see result in your terminal if there is no problem. (I will add webui sooner or later....)

## Feature
0. Ollama is the LLM interface! But you can switch to OpenAI
1. Support retrivel Rerank. Based on [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)
2. Support upload folder. You don't need to process file by file
3. Use Chroma as the local vectorDB
4. Support chat history load and store as JSON file

## TODO
- [ ] Support other chat history load and store, especially support similary search for chat history
- [ ] Support Stream output
- [ ] Add webui for better exprience
- [ ] Use LLamaIndex framework to refactor