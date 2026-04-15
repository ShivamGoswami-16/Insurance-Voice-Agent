from datetime import datetime, timezone
from pathlib import Path
import json
import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request 
from fastapi.responses import JSONResponse
from pyairtable import Api

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

app = FastAPI(title="Observe Claims Agent Backend")

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
CUSTOMERS_TABLE = os.getenv("AIRTABLE_CUSTOMERS_TABLE", "customers")
INTERACTIONS_TABLE = os.getenv("AIRTABLE_INTERACTIONS_TABLE", "interactions")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID:
    raise RuntimeError("Missing Airtable configuration in .env")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

airtable = Api(AIRTABLE_API_KEY)
customers_table = airtable.table(AIRTABLE_BASE_ID, CUSTOMERS_TABLE)
interactions_table = airtable.table(AIRTABLE_BASE_ID, INTERACTIONS_TABLE)

chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "app" / "chroma_db"))
embedding_fn = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

faq_collection = chroma_client.get_or_create_collection(
    name="faq_kb",
    embedding_function=embedding_fn,
)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


def normalize_phone(phone: str) -> str:
    return "".join(ch for ch in str(phone) if ch.isdigit())


def extract_tool_args(body: dict) -> dict:
    # direct flat JSON
    if "phone_number" in body or "question" in body or "summary" in body:
        return body

    # Vapi wrapped tool call: message -> toolCalls -> [0] -> function -> arguments
    tool_calls = body.get("message", {}).get("toolCalls", [])
    if tool_calls:
        args = tool_calls[0].get("function", {}).get("arguments")
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return {}

    # fallback wrapper
    args = body.get("function", {}).get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}

    return {}


@app.post("/lookup-customer")
async def lookup_customer(request: Request):
    body = await request.json()
    print("lookup-customer raw body:", body, flush=True)

    payload = extract_tool_args(body)

    tool_call_id = None
    tool_calls = body.get("message", {}).get("toolCalls", [])
    if tool_calls:
        tool_call_id = tool_calls[0].get("id")

    phone_number = payload.get("phone_number")

    if not phone_number:
        return JSONResponse(
            content={
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "error": "phone_number not found in request body"
                    }
                ]
            }
        )

    target = normalize_phone(phone_number)
    records = customers_table.all()

    for record in records:
        fields = record.get("fields", {})
        stored_phone = normalize_phone(fields.get("phone_number", ""))

        if stored_phone == target:
            response_dict = {
                "found": True,
                "first_name": fields.get("first_name"),
                "last_name": fields.get("last_name"),
                "claim_id": fields.get("claim_id"),
                "claim_status": fields.get("claim_status"),
                "documentation_required": fields.get("documentation_required", False),
                "documentation_instructions": fields.get(
                    "documentation_instructions",
                    "Please upload documents to the claims portal or email support@observeinsurance.com."
                ),
            }
            print("lookup-customer response:", response_dict, flush=True)

            return JSONResponse(
                content={
                    "results": [
                        {
                            "toolCallId": tool_call_id,
                            "result": json.dumps(response_dict, separators=(",", ":"))
                        }
                    ]
                }
            )

    return JSONResponse(
        content={
            "results": [
                {
                    "toolCallId": tool_call_id,
                    "result": json.dumps({"found": False}, separators=(",", ":"))
                }
            ]
        }
    )


@app.post("/faq-rag")
async def faq_rag(request: Request):
    body = await request.json()

    payload = extract_tool_args(body)

    tool_call_id = None
    tool_calls = body.get("message", {}).get("toolCalls", [])
    if tool_calls:
        tool_call_id = tool_calls[0].get("id")

    query = str(payload.get("question", "")).strip()

    if not query:
        return JSONResponse(
            content={
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "error": "question not found in request body"
                    }
                ]
            }
        )

    results = faq_collection.query(query_texts=[query], n_results=3)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    if not docs:
        response_dict = {
            "matches_found": False,
            "recommended_answer_context": "No relevant FAQ information found."
        }
        return JSONResponse(
            content={
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "result": json.dumps(response_dict, separators=(",", ":"))
                    }
                ]
            }
        )

    top_matches = []
    flat_parts = []

    for i, doc in enumerate(docs):
        title = metas[i].get("title", f"Match {i + 1}") if i < len(metas) and metas[i] else f"Match {i + 1}"
        score = distances[i] if i < len(distances) else None

        top_matches.append({
            "title": title,
            "content": doc,
            "distance": score,
        })
        flat_parts.append(f"{i + 1}. {title}: {doc}")

    response_dict = {
        "matches_found": True,
        "recommended_answer_context": "\n".join(flat_parts),
        "top_matches": top_matches,
    }

    return JSONResponse(
        content={
            "results": [
                {
                    "toolCallId": tool_call_id,
                    "result": json.dumps(response_dict, separators=(",", ":"))
                }
            ]
        }
    )


@app.post("/log-interaction")
async def log_interaction(request: Request):
    body = await request.json()

    payload = extract_tool_args(body)

    tool_call_id = None
    tool_calls = body.get("message", {}).get("toolCalls", [])
    if tool_calls:
        tool_call_id = tool_calls[0].get("id")

    caller_phone = payload.get("caller_phone", "")
    authenticated_name = payload.get("authenticated_name", "")
    summary = payload.get("summary", "")
    sentiment = payload.get("sentiment", "Neutral")
    needs_human = payload.get("needs_human", False)
    is_emergency = payload.get("is_emergency", False)
    outcome = payload.get("outcome", "completed")

    if not summary:
        return JSONResponse(
            content={
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "error": "summary not found in request body"
                    }
                ]
            }
        )

    record = interactions_table.create({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "caller_phone": caller_phone,
        "authenticated_name": authenticated_name,
        "summary": summary,
        "sentiment": sentiment,
        "needs_human": needs_human,
        "is_emergency": is_emergency,
        "outcome": outcome
    })

    response_dict = {
        "success": True,
        "record_id": record["id"]
    }

    return JSONResponse(
        content={
            "results": [
                {
                    "toolCallId": tool_call_id,
                    "result": json.dumps(response_dict, separators=(",", ":"))
                }
            ]
        }
    )


@app.post("/lookup-by-claim-id")
async def lookup_by_claim_id(request: Request):
    body = await request.json()

    payload = extract_tool_args(body)

    tool_call_id = None
    tool_calls = body.get("message", {}).get("toolCalls", [])
    if tool_calls:
        tool_call_id = tool_calls[0].get("id")

    claim_id = str(payload.get("claim_id", "")).strip()

    if not claim_id:
        return JSONResponse(
            content={
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "error": "claim_id not found in request body"
                    }
                ]
            }
        )

    records = customers_table.all()

    for record in records:
        fields = record.get("fields", {})
        stored_claim_id = str(fields.get("claim_id", "")).strip()

        if stored_claim_id.lower() == claim_id.lower():
            response_dict = {
                "found": True,
                "first_name": fields.get("first_name"),
                "last_name": fields.get("last_name"),
                "phone_number": fields.get("phone_number"),
                "claim_id": fields.get("claim_id"),
                "claim_status": fields.get("claim_status"),
                "documentation_required": fields.get("documentation_required", False),
                "documentation_instructions": fields.get(
                    "documentation_instructions",
                    "Please upload documents to the claims portal or email support@observeinsurance.com."
                ),
            }

            return JSONResponse(
                content={
                    "results": [
                        {
                            "toolCallId": tool_call_id,
                            "result": json.dumps(response_dict, separators=(",", ":"))
                        }
                    ]
                }
            )

    return JSONResponse(
        content={
            "results": [
                {
                    "toolCallId": tool_call_id,
                    "result": json.dumps({"found": False}, separators=(",", ":"))
                }
            ]
        }
    )