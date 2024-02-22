from chatpdf import ChatPDF
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    file_path = PROJECT_ROOT + '/test/I_have_a_dream.txt'
    chatpdf = ChatPDF(files_path=file_path, model_name="mistral:latest", rerank_top_k=4)

    print(
        chatpdf.chat(
            "what is the dream of Martin Luther King based on the reference data?"
        )
    )
