# ChatFile-langchain

The ChatFile project can parse various file types and use a vector database to retrieve questions, which are then answered by an LLM (Large Language Model).

I use Ollama as my LLM backend

## Feature
0. Ollama is the LLM interface! But you can switch to OpenAI
1. Support retrivel Rerank. Based on [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)
2. Support upload folder. You don't need to process file by file
3. Use Chroma as the local vectorDB

## TODO
- [ ] Support Stream output
- [ ] Add webui for better exprience