# Advanced-ChatFile

The Advanced-ChatFile project is designed to parse various file types and utilize a vector database for retrieving questions, which are subsequently answered by an LLM (Large Language Model).


This project serves as a practical implementation of the concepts discussed in the article [Mastering RAG: How To Architect An Enterprise RAG System](https://www.rungalileo.io/blog/mastering-rag-how-to-architect-an-enterprise-rag-system)

![image](/docs/architecture.png)


## NOTE!

This project is currently in development and may contain bugs and issues.

To explore the code or execute the project, simply place your data files into the 'test' folder for storage in the vectorDB.

Then, run `python main.py`. If everything is set up correctly, you will see the results in your terminal. (A web UI will be introduced in the future.)

### Get Started

First you should intall conda and create a environment and activate it. Then install pytorch based on [pytorch website](https://pytorch.org/get-started/locally/)

After that, run 
```
pip install -r requirements.txt
```

Remember rename `.env.example` to `.env` file name, this is the config file for the project!

Finally, run `python main.py`, you will see the test result.


## Advanced-RAG System Component

LLM Processing Layer:
- Query Rewriter
- Retrieval
- Improved Reranking
- General Output

Vector Database Storage Layer:
- Document Ingestion
- Document Storage

## Feature
1. OpenAI is the LLM interface, but you can switch to Ollama. And also use LM Studio as LLM interface, then you can use OpenAI-Like API.
2. Supports query rewriting, enhancing the original query's detail for improved processing by the LLM.
3. Implements reranking of retrieved documents to optimize relevance and accuracy. Based on [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)
4. Allows for uploading entire folders, eliminating the need to process files individually.
5. Utilizes Chroma as the local vector database.
6. Enables loading and storing chat history as a JSON file. 



## TODO
- [ ] Support loading and storing other chat histories, especially enabling similarity search for chat history.
- [ ] Support streaming output.
- [ ] Introduce a web UI for an enhanced experience.
- [ ] Refactor using the LLamaIndex framework.