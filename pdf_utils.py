import fitz  # PyMuPDF

def extract_text_from_pdf(file) -> list:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text_chunks = []
    for page in doc:
        text = page.get_text()
        if text:
            text_chunks.extend(text.split("\n\n"))
    return [chunk.strip() for chunk in text_chunks if chunk.strip()]
