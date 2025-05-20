import pickle
import logging

import faiss
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer

import config
from pdf_parser import extract_text_by_headers

# ------ Configuration ------
PDF_PATH = "Port Tariff.pdf"
INDEX_PATH = "data/port_tariff.index"
KEYS_PATH = "data/port_tariff_keys.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash-001"
TOP_K = 5  # number of chunks to retrieve

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

# Initialize the Gemini Developer API client
client = genai.Client(api_key=config.GEMINI_API_KEY)


def load_index_and_keys():
    """Load FAISS index and chunk keys."""
    logging.info(f"Loading FAISS index from {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    with open(KEYS_PATH, "rb") as f:
        keys = pickle.load(f)
    logging.info(f"Loaded {len(keys)} chunk keys")
    return index, keys


# Pre-load resources
chunks = extract_text_by_headers(PDF_PATH)
logging.info(f"Extracted {len(chunks)} header-based chunks from PDF")
embedder = SentenceTransformer(EMBED_MODEL_NAME)
index, keys = load_index_and_keys()


def retrieve_relevant_chunks(query, top_k=TOP_K):
    """
    Embed the query and retrieve the top_k similar chunks with distances.
    Returns a list of dicts: {'header', 'text', 'distance'}
    """
    # Embed query
    q_vec = embedder.encode(query).astype("float32")
    # Search index
    distances, indices = index.search(np.expand_dims(q_vec, axis=0), top_k)
    retrieved = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(keys):
            header = keys[idx]
            text = chunks.get(header, "")
            retrieved.append({"header": header, "text": text, "distance": float(dist)})
        else:
            logging.warning(f"Index returned out-of-range key idx={idx}")
    # Debug log
    for item in retrieved:
        logging.info(f"Retrieved chunk '{item['header']}' (dist={item['distance']:.4f})")
    return retrieved


def generate_answer(query, top_k=TOP_K):
    """
    RAG pipeline: retrieve context, call Gemini Developer API, return answer.
    """
    # Retrieve
    retrieved = retrieve_relevant_chunks(query, top_k)
    if not retrieved:
        raise ValueError("No relevant chunks found for the query.")

    # Build context
    context_sections = []
    for item in retrieved:
        header = item["header"]
        snippet = item["text"][:500] + ("..." if len(item["text"]) > 500 else "")  # preview
        context_sections.append(f"## {header}\n{snippet}")
    context = "\n\n".join(context_sections)

    # Compose prompt
    prompt = f"""
    You are an AI assistant specialized in maritime port tariffs.
    Use the following extracted sections to answer the user's query accurately.
    Return all amounts in South African Rand (ZAR) with a clear breakdown by tariff type.

    context: {context}
    User query: {query}
    """

    logging.info("Sending prompt to Gemini model...")
    # Call Gemini
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    answer = response.text
    logging.info("Received response from Gemini")
    return answer
