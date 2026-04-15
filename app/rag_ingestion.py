import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "faq_docs.json"

client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))

embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

collection = client.get_or_create_collection(
    name="faq_kb",
    embedding_function=embedding_fn,
)

with open(FAQ_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

ids = [d["id"] for d in docs]
documents = [f'{d["title"]}\n\n{d["text"]}' for d in docs]
metadatas = [{"title": d["title"]} for d in docs]

# reset-and-load pattern for simplicity
existing = collection.get()
if existing and existing.get("ids"):
    collection.delete(ids=existing["ids"])

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
)

print("Ingested", len(ids), "FAQ docs")