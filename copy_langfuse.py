from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pyairtable import Api
from typing import Any
from langfuse import get_client

CALL_STATE: dict[str, dict[str, Any]] = {}
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=r"C:\Work\observe-agent\app\.env")

app = FastAPI(title="Observe Claims Agent Backend")

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
CUSTOMERS_TABLE = os.getenv("AIRTABLE_CUSTOMERS_TABLE", "customers")
INTERACTIONS_TABLE = os.getenv("AIRTABLE_INTERACTIONS_TABLE", "interactions")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

langfuse = get_client()

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

def get_call_id_from_body(body: dict[str, Any]) -> str | None:
    return (
        body.get("call", {}).get("id")
        or body.get("message", {}).get("call", {}).get("id")
        or body.get("message", {}).get("artifact", {}).get("call", {}).get("id")
    )


def get_or_create_call_state(call_id: str) -> dict[str, Any]:
    if call_id not in CALL_STATE:
        trace = langfuse.start_observation(
            name="voice-call",
            as_type="span",
            input={"call_id": call_id},
            metadata={
                "source": "vapi",
                "service_name": "observe-agent",
                "environment": "dev",
                "version": "v1"
            }
        )

        CALL_STATE[call_id] = {
            "call_id": call_id,
            "trace_id": trace.trace_id,
            "trace": trace,
            "caller_phone": "",
            "authenticated_name": "",
            "customer_found": False,
            "claim_id": "",
            "claim_status": "",
            "faq_queries": [],
            "needs_human": False,
            "is_emergency": False,
            "summary": "",
            "sentiment": "neutral",
            "outcome": "in_progress",
            "transcript": "",
            "ended_reason": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    return CALL_STATE[call_id]


def normalize_phone(phone: str) -> str:
    return "".join(ch for ch in str(phone) if ch.isdigit())


def extract_tool_args(body: dict[str, Any]) -> dict[str, Any]:
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

    payload = extract_tool_args(body)
    phone_number = payload.get("phone_number")
    call_id = get_call_id_from_body(body)
    state = get_or_create_call_state(call_id) if call_id else None
    span = None

    if state:
        span = state["trace"].start_observation(
            name="lookup_customer",
            as_type="span",
            input={"phone_number": phone_number}
        )
        state["caller_phone"] = phone_number

    if not phone_number:
        raise HTTPException(status_code=400, detail="phone_number not found in request body")

    target = normalize_phone(phone_number)
    records = customers_table.all()

    for record in records:
        fields = record.get("fields", {})
        stored_phone = normalize_phone(fields.get("phone_number", ""))

        if stored_phone == target:
            if state:
                state["customer_found"] = True
                state["authenticated_name"] = f'{fields.get("first_name", "")} {fields.get("last_name", "")}'.strip()
                state["claim_id"] = fields.get("claim_id", "")
                state["claim_status"] = fields.get("claim_status", "")
                state["trace"].start_observation(
                    name="customer_lookup_success",
                    as_type="event",
                    input={"phone": phone_number}
                ).end()

            if span:
                span.update(
                    output={
                        "found": True,
                        "claim_status": fields.get("claim_status")
                    }
                )
                span.end()

            return {
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

    if state:
        state["customer_found"] = False
        state["trace"].start_observation(
            name="customer_lookup_failed",
            as_type="event",
            input={"phone": phone_number}
        ).end()

    if span:
        span.update(output={"found": False})
        span.end()

    return {"found": False}


@app.post("/faq-rag")
async def faq_rag(request: Request):
    body = await request.json()
    # print("faq-rag raw body:", body, flush=True)

    payload = extract_tool_args(body)
    query = str(payload.get("question", "")).strip()
    call_id = get_call_id_from_body(body)
    state = get_or_create_call_state(call_id) if call_id else None
    span = None

    if state:
        state["faq_queries"].append(query)
        state["trace"].start_observation(
            name="faq_query",
            as_type="event",
            input={"query": query}
        ).end()
        span = state["trace"].start_observation(
            name="faq_rag",
            as_type="span",
            input={"query": query}
        )

    if not query:
        raise HTTPException(status_code=400, detail="question not found in request body")

    results = faq_collection.query(query_texts=[query], n_results=3)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    if not docs:
        return {
            "matches_found": False,
            "recommended_answer_context": "No relevant FAQ information found."
        }

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

    if span:
        span.update(
            output={
                "question": query,
                "matches_found": True,
                "top_match_titles": [m["title"] for m in top_matches],
                "top_match_distances": [m["distance"] for m in top_matches],
            }
        )
        span.end()

    return {
        "matches_found": True,
        "recommended_answer_context": "\n".join(flat_parts),
        "top_matches": top_matches,
    }


@app.post("/log-interaction")
async def log_interaction(request: Request):
    body = await request.json()
    # print("log-interaction raw body:", body, flush=True)

    payload = extract_tool_args(body)
    call_id = get_call_id_from_body(body)

    if not call_id:
        return {"success": False, "message": "call_id not found; skipping state update"}

    state = get_or_create_call_state(call_id)
    if state:
        span = state["trace"].start_observation(
            name="log_interaction",
            as_type="span",
            input=payload
        )

        span.update(output={"status": "updated"})
        span.end()

    if payload.get("caller_phone"):
        state["caller_phone"] = payload.get("caller_phone", state["caller_phone"])

    if payload.get("authenticated_name"):
        state["authenticated_name"] = payload.get("authenticated_name", state["authenticated_name"])

    if payload.get("summary"):
        state["summary"] = payload["summary"]

    if payload.get("sentiment") in {"Positive", "Neutral", "Negative"}:
        state["sentiment"] = payload["sentiment"]

    state["needs_human"] = bool(payload.get("needs_human", state["needs_human"]))
    state["is_emergency"] = bool(payload.get("is_emergency", state["is_emergency"]))
    state["outcome"] = payload.get("outcome", state["outcome"])

    return {"success": True, "message": "call state updated"}

