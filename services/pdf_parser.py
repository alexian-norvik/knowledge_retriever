import re
import pickle
from pathlib import Path

import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from common import constants, llm_constants


def extract_text_by_headers(pdf_path):
    """
    Extract text from the PDF and split into chunks based on header lines.
    Headers are detected as lines starting with 'SECTION <number>' or all-caps lines of length >= 5.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    lines = text.splitlines()
    chunks = {}
    current_header = None
    for line in lines:
        line = line.strip()
        if re.match(constants.REGEX_PATTERN, line):
            current_header = line
            chunks[current_header] = []
        elif current_header:
            chunks[current_header].append(line)
    # Join lines into paragraphs
    return {h: "\n".join(p).strip() for h, p in chunks.items()}


def convert_chunks_to_markdown(chunks, output_path):
    """
    Convert header-based chunks to a Markdown file.
    """
    md = ""
    for header, content in chunks.items():
        md += f"## {header}\n\n{content}\n\n"
    Path(output_path).write_text(md)


def embed_chunks(chunks, model_name=llm_constants.EMBEDDING_MODEL):
    """
    Generate embeddings for each chunk using a SentenceTransformer model.
    """
    model = SentenceTransformer(model_name)
    embeddings = {header: model.encode(content) for header, content in chunks.items()}
    return embeddings


def build_faiss_index(embeddings, index_path, keys_path):
    """
    Build a FAISS index from embeddings and save the index along with the keys mapping.
    """
    dim = next(iter(embeddings.values())).shape[0]
    index = faiss.IndexFlatL2(dim)
    keys = list(embeddings.keys())
    matrix = np.vstack([embeddings[k] for k in keys]).astype("float32")
    index.add(matrix)
    faiss.write_index(index, index_path)
    with open(keys_path, "wb") as f:
        pickle.dump(keys, f)


def main(pdf_path: str) -> str:
    """
    run the functions one by one and return message that chunks are saved.
    """

    chunks = extract_text_by_headers(pdf_path)
    convert_chunks_to_markdown(chunks, constants.MARKDOWN_PATH)
    embeddings = embed_chunks(chunks)
    build_faiss_index(embeddings, constants.INDEX_PATH, constants.KEYS_PATH)

    return f"FAISS index saved to {constants.INDEX_PATH} with keys in {constants.KEYS_PATH}"
