"""
End-to-end test of AtomicRAG with real Gemini models.
Tests the full pipeline: Index (chunk -> extract -> embed -> graph) and Retrieve (anchor -> traverse -> rank).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomicrag import IndexPipeline, RetrievePipeline, AtomicRAGConfig
from atomicrag.integrations.gemini import GeminiLLM, GeminiEmbedding

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY environment variable to run this test.")

# Sample documents about a fictional tech company
SAMPLE_DOCUMENTS = [
    {
        "text": """NovaTech Cloud Platform v3.0 Release Notes

NovaTech Cloud Platform version 3.0 was released on January 15, 2026.
This release introduces several major features including native Kubernetes
orchestration, built-in service mesh powered by Istio, and zero-trust
security architecture. The platform now supports multi-region deployment
across AWS, Azure, and Google Cloud. NovaTech CTO Sarah Chen stated that
v3.0 represents a fundamental shift toward cloud-native infrastructure.
The pricing model has changed from per-VM licensing to consumption-based
billing. Enterprise customers can upgrade from v2.x with the automated
migration tool provided in the admin console.""",
        "doc_id": "novatech-release-notes",
    },
    {
        "text": """NovaTech vs CompeteStack: Enterprise Container Platform Comparison

When comparing NovaTech Cloud Platform with CompeteStack Enterprise,
several key differences emerge. NovaTech offers native Kubernetes support
while CompeteStack uses a proprietary container runtime called CStack.
In benchmarks conducted by CloudReview Labs in December 2025, NovaTech
demonstrated 40% faster pod scheduling and 25% lower memory overhead.
CompeteStack has stronger Windows container support and a more mature
CI/CD pipeline integration. NovaTech's pricing starts at $0.05 per
container-hour compared to CompeteStack's $0.08 per container-hour.
Both platforms support hybrid cloud deployments but NovaTech's multi-region
failover was rated "excellent" by Gartner while CompeteStack received "good".""",
        "doc_id": "novatech-competitive",
    },
    {
        "text": """NovaTech Security Architecture Whitepaper

NovaTech Cloud Platform implements a defense-in-depth security model.
All inter-service communication is encrypted using mutual TLS (mTLS).
The platform integrates with major identity providers including Okta,
Azure AD, and PingFederate for single sign-on. Role-based access control
(RBAC) follows the principle of least privilege with over 200 pre-defined
roles. The audit logging system captures all API calls and stores them
in tamper-proof storage for compliance. NovaTech achieved SOC 2 Type II
certification in November 2025 and FedRAMP authorization is expected
by Q2 2026. Vulnerability scanning is performed automatically on every
container image before deployment using the integrated NovaScan engine.""",
        "doc_id": "novatech-security",
    },
]

TEST_QUERIES = [
    "What are the main features of NovaTech Cloud Platform v3.0?",
    "How does NovaTech compare to CompeteStack in pricing?",
    "What security certifications does NovaTech have?",
]


def run_test():
    print("=" * 70)
    print("AtomicRAG End-to-End Test with Gemini")
    print("=" * 70)

    # Initialize models
    llm = GeminiLLM(api_key=API_KEY, model="gemini-2.5-flash")
    embedding = GeminiEmbedding(api_key=API_KEY, model="models/gemini-embedding-001")

    config = AtomicRAGConfig(
        chunk_size=800,
        chunk_overlap=100,
        verbose=True,
        result_top_n=3,
        traversal_depth=2,
        beam_size=10,
    )

    # === INDEX ===
    print("\n--- INDEXING ---\n")
    graph = IndexPipeline(llm=llm, embedding=embedding, config=config).run(SAMPLE_DOCUMENTS)

    print(f"\nGraph Stats: {graph.stats()}")
    print(f"Entities: {[e.name for e in graph.entities[:20]]}...")
    print(f"Sample KU: {graph.knowledge_units[0].content}")
    print(f"Embedding dim: {len(graph.knowledge_units[0].embedding)}")

    # Verify graph integrity
    assert graph.stats()["chunks"] >= 3, "Should have at least 3 chunks"
    assert graph.stats()["knowledge_units"] >= 5, "Should have multiple KUs"
    assert graph.stats()["entities"] >= 3, "Should have multiple entities"
    assert graph.stats()["edges"] >= 5, "Should have edges"
    assert len(graph.knowledge_units[0].embedding) > 0, "Embeddings should not be empty"
    print("\n[PASS] Graph integrity checks passed")

    # === SAVE / LOAD ===
    print("\n--- SAVE / LOAD ---\n")
    graph.to_json("/tmp/atomicrag_test_graph.json")
    from atomicrag.models.graph import KnowledgeGraph

    loaded = KnowledgeGraph.from_json("/tmp/atomicrag_test_graph.json")
    assert loaded.stats() == graph.stats(), "JSON roundtrip failed"
    print("[PASS] JSON save/load roundtrip OK")

    # === RETRIEVE ===
    print("\n--- RETRIEVAL ---\n")
    retriever = RetrievePipeline(graph=graph, llm=llm, embedding=embedding, config=config)

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = retriever.search(query)

        print(f"Entities extracted: {results.entities_extracted}")
        print(f"Graph stats: {results.graph_stats}")
        print(f"Results: {len(results.items)} items")

        for i, item in enumerate(results.items, 1):
            print(f"  {i}. score={item.score:.3f} | entities={item.entity_names[:5]}")
            print(f"     {item.content[:120]}...")

        assert len(results.items) > 0, f"Should return results for: {query}"
        assert results.items[0].score > 0, "Top result should have positive score"
        print(f"[PASS] Query returned {len(results.items)} results")

    # === RESULT EXPORT ===
    print("\n--- RESULT EXPORT ---\n")
    last_result = results
    json_str = last_result.to_json()
    assert '"query"' in json_str
    assert '"items"' in json_str
    print(f"[PASS] JSON export: {len(json_str)} chars")

    d = last_result.to_dict()
    assert "items" in d
    assert "entities_extracted" in d
    print("[PASS] Dict export OK")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
