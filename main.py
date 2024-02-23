from chatpdf import ChatPDF
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    file_path = PROJECT_ROOT + '/test'
    chatpdf = ChatPDF(files_path=file_path, model_name="mistral:latest", rerank_top_k=3)

    resp, ref = chatpdf.predict("使用 langchain 实现了什么本地程序？")
    print(resp, '\n\n', ref)

    # 更新文件
    file2 = PROJECT_ROOT + "/README.md"
    chatpdf.add_single_file(file2)
    resp, ref = chatpdf.predict("使用 langchain 实现了什么本地程序？")
    print(resp, '\n\n', ref)

    # 已有向量数据库，测试是否能够准确召回信息
    # chatpdf = ChatPDF()
    # resp, ref = chatpdf.predict("使用 langchain 实现了什么本地程序？")
    # print(resp, '\n\n', ref)