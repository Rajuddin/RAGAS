"""
rag_client.py
Utility to call the RAG system API and extract:
  - generated_answer : the LLM-generated response
  - contexts         : list of retrieved document chunks
"""

import json
import re
import requests
from typing import Tuple, List, Optional
from config_loader import RAGApiConfig

# Field names tried (in order) when auto-detecting the answer / contexts
# fields in an API response that wasn't given explicit overrides.
ANSWER_FIELD_CANDIDATES = [
    "answer", "generated_answer", "result", "output_text", "response", "text",
]
CONTEXTS_FIELD_CANDIDATES = [
    "contexts", "retrieved_contexts", "retrieved_docs", "retrieved_doc",
    "source_documents", "input_documents", "documents", "context",
]
# Key tried (in order) to pull text out of a context item that's a dict
# rather than a plain string, e.g. {"page_content": "..."}.
CONTEXT_ITEM_FIELD_CANDIDATES = ["page_content", "content", "text", "chunk"]


def _first_present(data: dict, candidates: List[str]) -> Optional[str]:
    return next((key for key in candidates if key in data), None)


_PATH_SEGMENT_RE = re.compile(r"[^.\[\]]+")


def _resolve_path(data, path: str):
    """Look up a value in a nested dict/list response using a dotted path that
    may include array indices, e.g. "message.generated_response" or
    "message.context[0].context". Returns None if any segment is missing."""
    current = data
    for part in _PATH_SEGMENT_RE.findall(path):
        if isinstance(current, list):
            if not part.lstrip("-").isdigit():
                return None
            idx = int(part)
            if not (-len(current) <= idx < len(current)):
                return None
            current = current[idx]
        elif isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        else:
            return None
    return current


def _get_field(data: dict, field_override: Optional[str], candidates: List[str]):
    """Resolve a response field: a nested/array path when `field_override` is
    set (see _resolve_path), otherwise the first top-level key auto-detected
    from `candidates`."""
    if field_override:
        return _resolve_path(data, field_override)
    key = _first_present(data, candidates)
    return data.get(key) if key else None


def _extract_context_text(item, item_field: Optional[str] = None) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if item_field:
            return str(item.get(item_field, item))
        key = _first_present(item, CONTEXT_ITEM_FIELD_CANDIDATES)
        if key:
            return str(item[key])
    return str(item)


def _coerce_static_value(raw: str):
    """Parse a hardcoded payload value as JSON when possible (so "123" -> 123,
    "true" -> True, '["a","b"]' -> a list), falling back to the raw string
    (so "text" stays the plain string "text")."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def build_payload(item: dict, payload_params: List[dict]) -> dict:
    """Build the RAG API request body from a test case + the configured payload
    parameters (see RAGApiConfig.payload_params): each entry is either a value
    pulled from a field of `item` (varies per test case) or a hardcoded value
    (the same for every test case)."""
    payload = {}
    for param in payload_params or []:
        name = (param.get("name") or "").strip()
        if not name:
            continue
        if param.get("source") == "static":
            payload[name] = _coerce_static_value(param.get("value", ""))
        else:
            payload[name] = item.get((param.get("field") or "").strip(), "")
    return payload


def query_rag_system(
    item: dict,
    rag_cfg: RAGApiConfig,
) -> Tuple[str, List[str]]:
    """
    Call the RAG API endpoint for one test case and extract (generated_answer, contexts).

    `item` is the test case dict (e.g. {"query": ..., "ground_truth": ...} plus
    any other fields present in the test data JSON); the request payload is
    built from it according to rag_cfg.payload_params (see build_payload / the
    UI's "Request Payload" editor in the RAG API Configuration panel).

    Field names are auto-detected from ANSWER_FIELD_CANDIDATES /
    CONTEXTS_FIELD_CANDIDATES / CONTEXT_ITEM_FIELD_CANDIDATES above, which
    cover the common shapes (answer/contexts, result/source_documents,
    output_text/input_documents, answer/retrieved_docs, etc).

    If your API uses field names outside those lists, set
    rag_cfg.answer_field / contexts_field / context_item_field (exposed in
    the UI's "RAG API Configuration" panel) to override detection.
    """
    payload = build_payload(item, rag_cfg.payload_params)

    response = requests.post(
        url=rag_cfg.endpoint,
        headers=rag_cfg.headers,
        json=payload,
        timeout=rag_cfg.timeout,
    )
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        # raise_for_status()'s message alone only has the status code, not the
        # API's actual validation error, which is almost always in the body
        # (e.g. "field 'question' is required") — surface it so payload
        # mismatches (see build_payload / the Request Payload editor) are
        # diagnosable from the UI error message instead of a bare "400".
        body = response.text.strip()
        detail = f" — response body: {body[:1000]}" if body else " — response body was empty"
        raise requests.exceptions.HTTPError(f"{exc}{detail}", response=response) from exc
    data = response.json()

    generated_answer = _get_field(data, rag_cfg.answer_field, ANSWER_FIELD_CANDIDATES)
    raw_contexts = _get_field(data, rag_cfg.contexts_field, CONTEXTS_FIELD_CANDIDATES)

    if generated_answer is None or raw_contexts is None:
        raise ValueError(
            f"Unexpected RAG API response structure. Got top-level keys: {list(data.keys())}. "
            "Set the Answer Field / Contexts Field overrides in the RAG API Configuration panel "
            "to match your API's response — nested responses are supported via a dotted path, "
            "e.g. \"message.generated_response\" or \"message.context[0].context\"."
        )

    if not isinstance(raw_contexts, list):
        raw_contexts = [raw_contexts]

    contexts = [_extract_context_text(item, rag_cfg.context_item_field) for item in raw_contexts]

    return generated_answer, contexts
