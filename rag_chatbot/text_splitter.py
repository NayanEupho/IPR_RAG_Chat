from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Tuple

import nltk
from nltk.tokenize import sent_tokenize

# Ensure tokenizer is available (downloads once)
nltk.download("punkt", quiet=True)


def split_text_into_chunks(
    pages: List[Tuple[str, str, int, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Splits cleaned text (with metadata) into chunks using LangChain's splitter.
    Adds section titles, chunk type (text/table), and sentence-level metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []

    for text, source_file, page_num, section in pages:
        if "[Table Data]" in text:
            parts = text.split("[Table Data]", 1)
            main_text = parts[0].strip()
            table_text = "[Table Data]" + parts[1].strip() if len(parts) > 1 else ""

            # Split main text into chunks
            if main_text:
                docs = splitter.create_documents([main_text])
                for i, doc in enumerate(docs):
                    doc.metadata = {
                        "source": source_file,
                        "page": page_num,
                        "chunk": i + 1,
                        "type": "text",
                        "section": section,
                        "sentences": sent_tokenize(doc.page_content)
                    }
                    all_chunks.append(doc)

            # Keep table chunk whole
            if table_text:
                doc = Document(
                    page_content=table_text,
                    metadata={
                        "source": source_file,
                        "page": page_num,
                        "chunk": len(all_chunks) + 1,
                        "type": "table",
                        "section": section,
                        "sentences": sent_tokenize(table_text)
                    }
                )
                all_chunks.append(doc)
        else:
            # Standard non-table chunking
            docs = splitter.create_documents([text])
            for i, doc in enumerate(docs):
                doc.metadata = {
                    "source": source_file,
                    "page": page_num,
                    "chunk": i + 1,
                    "type": "text",
                    "section": section,
                    "sentences": sent_tokenize(doc.page_content)
                }
                all_chunks.append(doc)

    return all_chunks
