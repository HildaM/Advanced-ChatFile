"""
自己实现不同文件的解析逻辑
"""

@staticmethod
def from_pdf(file_path: str):
    import PyPDF2

    contents = []
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text().strip()
            raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
            new_text = ""

            for text in raw_text:
                new_text += text
                # 一句一句地获取
                if text[-1] in [
                    ".",
                    "!",
                    "?",
                    "。",
                    "！",
                    "？",
                    "…",
                    ";",
                    "；",
                    ":",
                    "：",
                    "”",
                    "’",
                    "）",
                    "】",
                    "》",
                    "」",
                    "』",
                    "〕",
                    "〉",
                    "》",
                    "〗",
                    "〞",
                    "〟",
                    "»",
                    '"',
                    "'",
                    ")",
                    "]",
                    "}",
                ]:
                    contents.append(new_text)
                    new_text = ""

            if new_text:
                contents.append(new_text)

        return contents


@staticmethod
def from_txt(file_path: str):
    """Extract text content from a TXT file."""
    with open(file_path, "r", encoding="utf-8") as f:
        contents = [text.strip() for text in f.readlines() if text.strip()]
    return contents


@staticmethod
def from_docx(file_path: str):
    """Extract text content from a DOCX file."""
    import docx

    document = docx.Document(file_path)
    contents = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]
    return contents


@staticmethod
def from_markdown(file_path: str):
    """Extract text content from a Markdown file."""
    import markdown
    from bs4 import BeautifulSoup

    with open(file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    contents = [text.strip() for text in soup.get_text().splitlines() if text.strip()]
    return contents
