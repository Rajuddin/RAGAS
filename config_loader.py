"""
config_loader.py
Loads and validates configuration from config.yaml
"""

import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Payload sent to the RAG API when the request payload isn't customized (see
# RAGApiConfig.payload_params below): {"question": <query>, "chat_history": []}.
DEFAULT_PAYLOAD_PARAMS = [
    {"name": "question", "source": "field", "field": "query", "value": ""},
    {"name": "chat_history", "source": "static", "field": "", "value": "[]"},
]


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    api_version: Optional[str] = None
    embedding_model_name: Optional[str] = None


@dataclass
class RAGApiConfig:
    endpoint: str
    timeout: int
    headers: dict
    # Optional overrides for APIs whose response JSON doesn't match any
    # auto-detected shape (see rag_client.py). Leave blank to auto-detect.
    answer_field: Optional[str] = None
    contexts_field: Optional[str] = None
    context_item_field: Optional[str] = None
    # Defines the JSON request body sent to the RAG API. Each entry is either
    # {"name": <payload key>, "source": "field", "field": <key in the test case
    # JSON to pull the value from>} for a value that varies per test case, or
    # {"name": <payload key>, "source": "static", "value": <literal>} for a
    # value that's the same across every test case (see rag_client.build_payload).
    payload_params: List[dict] = field(default_factory=lambda: [dict(p) for p in DEFAULT_PAYLOAD_PARAMS])


@dataclass
class AppConfig:
    llm: LLMConfig
    rag_api: RAGApiConfig
    ragas_metrics: list
    allure_results_dir: str


def load_config(config_path: str = None) -> AppConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    provider = raw.get("llm_provider", "openai").lower()

    if provider == "azure":
        az = raw["azure"]
        llm_cfg = LLMConfig(
            provider="azure",
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", az.get("api_key", "")),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", az.get("azure_endpoint", "")),
            deployment_name=az.get("deployment_name", ""),
            embedding_model_name=az.get("embedding_model_name", "text-embedding-ada-002"),
            api_version=az.get("api_version", "2024-02-01"),
        )
    elif provider == "foundry":
        fo = raw["foundry"]
        llm_cfg = LLMConfig(
            provider="foundry",
            api_key=os.environ.get("AZURE_AI_FOUNDRY_API_KEY", fo.get("api_key", "")),
            azure_endpoint=os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT", fo.get("endpoint", "")),
            deployment_name=fo.get("model_name", ""),
            embedding_model_name=fo.get("embedding_model_name", ""),
            api_version=fo.get("api_version", "2024-05-01-preview"),
        )
    else:
        oa = raw["openai"]
        llm_cfg = LLMConfig(
            provider="openai",
            api_key=os.environ.get("OPENAI_API_KEY", oa.get("api_key", "")),
            model=oa.get("model", "gpt-4o"),
        )

    rag_cfg = RAGApiConfig(
        endpoint=os.environ.get("RAG_API_ENDPOINT", raw["rag_api"]["endpoint"]),
        timeout=raw["rag_api"].get("timeout", 30),
        headers=raw["rag_api"].get("headers", {"Content-Type": "application/json"}),
        answer_field=raw["rag_api"].get("answer_field") or None,
        contexts_field=raw["rag_api"].get("contexts_field") or None,
        context_item_field=raw["rag_api"].get("context_item_field") or None,
        payload_params=raw["rag_api"].get("payload_params") or [dict(p) for p in DEFAULT_PAYLOAD_PARAMS],
    )

    return AppConfig(
        llm=llm_cfg,
        rag_api=rag_cfg,
        ragas_metrics=raw["ragas"].get("metrics", []),
        allure_results_dir=raw["allure"].get("results_dir", "./allure-results"),
    )


def save_llm_config(
    provider: str,
    openai_fields: Optional[dict] = None,
    azure_fields: Optional[dict] = None,
    foundry_fields: Optional[dict] = None,
    config_path: str = None,
) -> None:
    """Persist LLM provider + field values into config.yaml, leaving other sections untouched."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw["llm_provider"] = provider
    if openai_fields:
        raw.setdefault("openai", {}).update(openai_fields)
    if azure_fields:
        raw.setdefault("azure", {}).update(azure_fields)
    if foundry_fields:
        raw.setdefault("foundry", {}).update(foundry_fields)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)


def save_rag_api_config(
    endpoint: str,
    timeout: int,
    headers: dict,
    answer_field: Optional[str] = None,
    contexts_field: Optional[str] = None,
    context_item_field: Optional[str] = None,
    payload_params: Optional[List[dict]] = None,
    config_path: str = None,
) -> None:
    """Persist RAG API endpoint/timeout/headers into config.yaml, leaving other sections untouched."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw["rag_api"] = {
        "endpoint": endpoint,
        "timeout": timeout,
        "headers": headers,
        "answer_field": answer_field or "",
        "contexts_field": contexts_field or "",
        "context_item_field": context_item_field or "",
        "payload_params": payload_params or [dict(p) for p in DEFAULT_PAYLOAD_PARAMS],
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)


def _is_foundry_v1_endpoint(endpoint: Optional[str]) -> bool:
    return "/openai/v1" in (endpoint or "")


def _build_foundry_v1_chat(llm_cfg: LLMConfig):
    """Azure AI Foundry's OpenAI v1-compatible API: plain OpenAI client + custom base_url.

    Auth and versioning for this API use the `api-key` header and an `api-version`
    query param (often the literal value "preview"), not OAuth Bearer + path-based
    deployment routing like the classic Azure OpenAI API.
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=llm_cfg.api_key,
        base_url=llm_cfg.azure_endpoint,
        model=llm_cfg.deployment_name,
        temperature=0,
        default_headers={"api-key": llm_cfg.api_key},
        default_query={"api-version": llm_cfg.api_version or "preview"},
    )


def _build_foundry_v1_embeddings(llm_cfg: LLMConfig):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        api_key=llm_cfg.api_key,
        base_url=llm_cfg.azure_endpoint,
        model=llm_cfg.embedding_model_name or "text-embedding-ada-002",
        default_headers={"api-key": llm_cfg.api_key},
        default_query={"api-version": llm_cfg.api_version or "preview"},
    )


def build_langchain_llm(llm_cfg: LLMConfig):
    """Build a Ragas-wrapped LLM object based on provider config.

    Wrapping with LangchainLLMWrapper (instead of returning the raw LangChain chat
    model) routes RAGAS's structured-output parsing through its markdown-fence-safe
    JSON extractor with built-in self-correction retry. Passing a raw LangChain
    BaseLanguageModel directly to a metric skips that path and crashes outright the
    first time the model wraps its JSON answer in a ```json code fence.
    """
    import warnings
    from ragas.llms import LangchainLLMWrapper

    if llm_cfg.provider == "azure":
        if _is_foundry_v1_endpoint(llm_cfg.azure_endpoint):
            raw_llm = _build_foundry_v1_chat(llm_cfg)
        else:
            from langchain_openai import AzureChatOpenAI
            raw_llm = AzureChatOpenAI(
                openai_api_key=llm_cfg.api_key,
                azure_endpoint=llm_cfg.azure_endpoint,
                azure_deployment=llm_cfg.deployment_name,
                openai_api_version=llm_cfg.api_version,
                temperature=0,
            )
    elif llm_cfg.provider == "foundry":
        if _is_foundry_v1_endpoint(llm_cfg.azure_endpoint):
            raw_llm = _build_foundry_v1_chat(llm_cfg)
        else:
            # Azure AI Foundry serverless / Models-as-a-Service: azure-ai-inference SDK.
            from azure.core.credentials import AzureKeyCredential
            from langchain_azure_ai.chat_models.inference import AzureAIChatCompletionsModel
            raw_llm = AzureAIChatCompletionsModel(
                endpoint=llm_cfg.azure_endpoint,
                credential=AzureKeyCredential(llm_cfg.api_key),
                model_name=llm_cfg.deployment_name,
                api_version=llm_cfg.api_version,
                temperature=0,
            )
    else:
        from langchain_openai import ChatOpenAI
        raw_llm = ChatOpenAI(
            openai_api_key=llm_cfg.api_key,
            model=llm_cfg.model,
            temperature=0,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(raw_llm)


def build_langchain_embeddings(llm_cfg: LLMConfig):
    """Build a Ragas-wrapped Embeddings object based on provider config."""
    import warnings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if llm_cfg.provider == "azure":
        if _is_foundry_v1_endpoint(llm_cfg.azure_endpoint):
            raw_embeddings = _build_foundry_v1_embeddings(llm_cfg)
        else:
            from langchain_openai import AzureOpenAIEmbeddings
            raw_embeddings = AzureOpenAIEmbeddings(
                openai_api_key=llm_cfg.api_key,
                azure_endpoint=llm_cfg.azure_endpoint,
                azure_deployment=llm_cfg.embedding_model_name or "text-embedding-ada-002",
                openai_api_version=llm_cfg.api_version,
            )
    elif llm_cfg.provider == "foundry":
        if _is_foundry_v1_endpoint(llm_cfg.azure_endpoint):
            raw_embeddings = _build_foundry_v1_embeddings(llm_cfg)
        else:
            from azure.core.credentials import AzureKeyCredential
            from langchain_azure_ai.embeddings.inference import AzureAIEmbeddingsModel
            raw_embeddings = AzureAIEmbeddingsModel(
                endpoint=llm_cfg.azure_endpoint,
                credential=AzureKeyCredential(llm_cfg.api_key),
                model_name=llm_cfg.embedding_model_name,
                api_version=llm_cfg.api_version,
            )
    else:
        from langchain_openai import OpenAIEmbeddings
        raw_embeddings = OpenAIEmbeddings(openai_api_key=llm_cfg.api_key)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainEmbeddingsWrapper(raw_embeddings)
