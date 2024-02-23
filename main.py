from chatpdf import ChatPDF
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    file_path = PROJECT_ROOT + '/test'
    chatpdf = ChatPDF(files_path=file_path, model_name="mistral:latest", rerank_top_k=4)

    resp, ref = chatpdf.predict("What is Martin Luther King say mainly about?")
    print(resp, '\n\n', ref)