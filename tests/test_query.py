"""Query endpoint tests with a fake chat model and fake embeddings."""

from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from app.retrieval import INSUFFICIENT_MESSAGE


def _ingest(client: TestClient, pdf: bytes) -> None:
    res = client.post("/ingest/pdf", files=[("files", ("crops.pdf", pdf, "application/pdf"))])
    assert res.json()["results"][0]["status"] == "ingested"


def test_health(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["index_chunks"] == 0


def test_query_on_empty_index_returns_409(client: TestClient):
    res = client.post("/query", json={"question": "When is wheat sown?"})
    assert res.status_code == 409
    assert "empty" in res.json()["detail"]


def test_query_returns_grounded_answer_with_citations(make_client, sample_pdf: bytes):
    client = make_client(responses=["Wheat is sown from late October to November [1]."])
    _ingest(client, sample_pdf)

    res = client.post("/query", json={"question": "When is wheat sown?", "top_k": 2})
    assert res.status_code == 200
    body = res.json()
    assert body["answerable"] is True
    assert "[1]" in body["answer"]
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["ref"] == 1
    assert citation["source"] == "crops.pdf"
    assert citation["page"] in (1, 3)
    assert citation["snippet"]


def test_query_prompt_contains_only_retrieved_context(make_client, sample_pdf: bytes):
    seen: dict = {}

    def capture(prompt_value):
        seen["messages"] = prompt_value.to_messages()
        return "Apply 120 kg nitrogen per hectare [1][2]."

    client = make_client(llm=RunnableLambda(capture))
    _ingest(client, sample_pdf)
    body = client.post("/query", json={"question": "How much nitrogen for wheat?"}).json()

    system, human = seen["messages"]
    assert "ONLY" in system.content and "INSUFFICIENT_CONTEXT" in system.content
    assert "Source: crops.pdf, page" in human.content
    assert [c["ref"] for c in body["citations"]] == [1, 2]


def test_query_insufficient_context(make_client, sample_pdf: bytes):
    client = make_client(responses=["INSUFFICIENT_CONTEXT"])
    _ingest(client, sample_pdf)
    body = client.post("/query", json={"question": "What is the price of tractors in Brazil?"}).json()
    assert body["answerable"] is False
    assert body["answer"] == INSUFFICIENT_MESSAGE
    assert body["citations"] == []


def test_uncited_answer_is_withheld_by_guardrail(make_client, sample_pdf: bytes):
    client = make_client(responses=["Transplant paddy at 21 to 25 days."])
    _ingest(client, sample_pdf)
    body = client.post("/query", json={"question": "When to transplant paddy?", "top_k": 2}).json()
    assert body["answerable"] is False
    assert body["guardrail"] == "no_citation"
    assert body["citations"] == [] and body["warnings"]


def test_uncited_answer_allowed_when_guardrail_off(make_client, settings, sample_pdf: bytes):
    settings.guardrail_require_citations = False
    client = make_client(responses=["Transplant paddy at 21 to 25 days."])
    _ingest(client, sample_pdf)
    body = client.post("/query", json={"question": "When to transplant paddy?", "top_k": 2}).json()
    assert body["answerable"] is True
    assert len(body["citations"]) == 2 and body["warnings"]


def test_query_llm_failure_returns_502(make_client, sample_pdf: bytes):
    def fail(_):
        raise ConnectionError("provider down")

    client = make_client(llm=RunnableLambda(fail))
    _ingest(client, sample_pdf)
    res = client.post("/query", json={"question": "When is wheat sown?"})
    assert res.status_code == 502
    assert "ConnectionError" in res.json()["detail"] and "connect" in res.json()["detail"]


def test_query_validation(client: TestClient):
    assert client.post("/query", json={"question": "  "}).status_code == 422
    assert client.post("/query", json={"question": "Wheat?", "top_k": 0}).status_code == 422
    assert client.post("/query", json={}).status_code == 422

