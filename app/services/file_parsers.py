import io
import pdfplumber


def parse_pdf(content: bytes) -> str:
    texts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

    if not texts:
        raise RuntimeError("No text extracted from PDF")

    return "\n\n".join(texts)

def parse_text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("utf-8", errors="ignore")

def parse_file(content: bytes, file_type: str) -> str:
    ft = file_type.lower()

    if ft == "pdf":
        return parse_pdf(content)
    elif ft in ("md", "markdown", "txt"):
        return parse_text(content)
    else:
        raise RuntimeError(f"Unsupported file type: {file_type}")
