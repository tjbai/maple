import PyPDF2 as pdf
from typing import List
import numpy as np
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from tqdm import tqdm
import pickle
import pinecone
from uuid import uuid4
from datetime import datetime

tiktoken.encoding_for_model("gpt-3.5-turbo")

DIMENSION = 1536
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "prob"
PINECONE_ENV = "asia-southeast1-gcp-free"
MODEL = "text-embedding-ada-002"


def token_length(chunk: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(chunk, disallowed_special=())
    return len(tokens)


def read_pdf(path: str) -> List[str]:
    file = open(path, "rb")
    pdfReader = pdf.PdfReader(file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=token_length,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for page in tqdm(pdfReader.pages):
        text = page.extract_text()
        chunks.extend(text_splitter.split_text(text))

    return chunks


def compute_embeddings(
    chunks: List[str], pickle_path: str = None
) -> np.ndarray[np.ndarray]:
    encoder = OpenAIEmbeddings(model=MODEL, openai_api_key=OPENAI_API_KEY)
    embds = encoder.embed_documents(chunks)
    print(f"\n>>> generated embeddings {datetime.now()}")

    embeddings = np.array([np.array(e) for e in embds])
    if pickle_path:
        with open(pickle_path, "wb") as f:
            pickle.dump(embeddings, f)

    return embeddings


def push_to_pinecone(
    chunks: List[str],
    embeddings: np.ndarray[np.ndarray] = None,
    pickle_path: str = None,
):
    if pickle_path:
        with open(pickle_path, "rb") as f:
            embeddings = pickle.load(f)
    embeddings = list([list(e) for e in embeddings])

    assert embeddings is not None
    assert len(embeddings) == len(chunks)

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=INDEX_NAME, metric="cosine", dimension=DIMENSION)

    index = pinecone.Index(INDEX_NAME)  # use GRPCIndex for multi-pod setups
    print(f"\n>>> obtained index {datetime.now()}")

    ids = [str(uuid4()) for _ in range(len(embeddings))]
    metadatas = [{"index": i, "chunk": chunk} for i, chunk in enumerate(chunks)]
    vectors = [
        (id, embedding, metadata)
        for id, embedding, metadata in zip(ids, embeddings, metadatas)
    ]

    for i in tqdm(range(0, len(vectors), 100)):
        slice = vectors[i : i + 100]
        index.upsert(vectors=slice)


if __name__ == "__main__":
    chunks = read_pdf("../data/prob.pdf")
    # embds = compute_embeddings(chunks, pickle_path="../data/pickles/prob.pkl")
    push_to_pinecone(chunks, pickle_path="../data/pickles/prob.pkl")
