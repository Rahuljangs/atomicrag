"""
Test all 3 extraction methods: llm, vocabulary, sentence.
Compares output quality and speed.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomicrag import IndexPipeline, RetrievePipeline, AtomicRAGConfig
from atomicrag.integrations.gemini import GeminiLLM, GeminiEmbedding

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY environment variable to run this test.")

SAMPLE_DOC = """
Red Hat OpenShift Developer Services

Products that allow developers to quickly and securely deliver applications to production.
They enable platform engineers to more easily enforce operational controls and compliance
across hybrid multi-cloud environments.

Internal Developer Portal: Red Hat Developer Hub provides a centralized portal for developers
to discover APIs, services, and documentation. It integrates with Backstage for plugin support.

Kube Developer Desktop Experience: Podman Desktop provides a local container development
experience without requiring root privileges. It supports Docker-compatible workflows.

Migration Services: Red Hat Migration Toolkit for Applications assists with migrating
and modernizing applications to Red Hat OpenShift and Application Runtimes. The Migration
Toolkit for Virtualization helps migrate VMs to OpenShift Virtualization. The Migration
Toolkit for Containers helps migrate containers between OpenShift clusters.

IDE Tools: Red Hat IDE Plugins provide language support for Java, Quarkus, and Kubernetes.
Red Hat Dev Spaces offers cloud-based development environments powered by Eclipse Che.

Trusted Software Supply Chain: Red Hat Trusted Application Pipeline automates build
verification and SBOM generation. Red Hat Trusted Artifact Signer provides keyless
signing tied to OIDC Identity. Red Hat Trusted Profile Analyzer monitors CVEs and
security posture. All inter-service communication is encrypted using mutual TLS.
NovaTech achieved SOC 2 Type II certification in November 2025.

Application Networking: Red Hat Service Interconnect enables secure multi-cluster
service connectivity across hybrid cloud environments using Skupper.

Konveyor is a CNCF sandbox project helping modernize applications to Kubernetes.
Contributors include Red Hat, IBM Research, and claranet. The Konveyor Hub process
includes Planning, Assessment, Analysis, Assets Generation, Reporting, and Code Suggestions.

The SPACE framework measures developer productivity across five dimensions:
Satisfaction and Wellbeing, Performance, Activity, Communication and Collaboration,
and Efficiency and Flow. This was introduced in 2021 as an improvement over the
DORA metrics from the 2014 State of DevOps Report.

Modern development workflows involve an Inner Loop (desktop development with writing
code, building, pushing, and debugging) and an Outer Loop (at-scale operations with
automated builds, validation tests, deployment, security, and compliance checks).
Platform Engineers are described as the Developer's Developer.
"""

QUERY = "What are the Red Hat OpenShift Developer Services?"


def run_method(method_name, doc_text, llm, embedding):
    print(f"\n{'=' * 60}")
    print(f"  METHOD: {method_name}")
    print(f"{'=' * 60}")

    config = AtomicRAGConfig(
        chunk_size=1000,
        chunk_overlap=150,
        ku_extraction_method=method_name,
        verbose=True,
        result_top_n=3,
        traversal_depth=2,
        vocab_min_term_freq=2,
        vocab_max_terms_per_llm_call=500,
    )

    # Index
    start = time.time()
    graph = IndexPipeline(llm=llm, embedding=embedding, config=config).run(
        [{"text": doc_text, "doc_id": "app-mod"}]
    )
    index_time = time.time() - start

    stats = graph.stats()
    print(f"\n  Stats: {stats}")
    print(f"  Index time: {index_time:.1f}s")
    print(f"  Entities: {[e.name for e in graph.entities[:10]]}...")

    assert stats["knowledge_units"] > 0, f"No KUs for method {method_name}"
    assert stats["entities"] > 0, f"No entities for method {method_name}"

    # Retrieve
    start = time.time()
    results = RetrievePipeline(graph=graph, llm=llm, embedding=embedding, config=config).search(
        QUERY
    )
    retrieve_time = time.time() - start

    print(f'\n  Query: "{QUERY}"')
    print(f"  Retrieve time: {retrieve_time:.1f}s")
    print(f"  Results: {len(results.items)}")

    for i, item in enumerate(results.items, 1):
        content_preview = item.content.replace("\n", " ")[:100]
        print(f"    [{i}] score={item.score:.3f} | {content_preview}...")

    assert len(results.items) > 0, f"No results for method {method_name}"
    print(f"\n  [PASS] {method_name} method works!")

    return {
        "method": method_name,
        "kus": stats["knowledge_units"],
        "entities": stats["entities"],
        "edges": stats["edges"],
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "top_score": results.items[0].score if results.items else 0,
    }


def main():
    print("=" * 60)
    print("EXTRACTION METHOD COMPARISON")
    print("=" * 60)

    doc_text = SAMPLE_DOC
    llm = GeminiLLM(api_key=API_KEY, model="gemini-2.5-flash")
    embedding = GeminiEmbedding(api_key=API_KEY, model="models/gemini-embedding-001")

    results = []

    # Method 1: sentence (zero LLM cost)
    results.append(run_method("sentence", doc_text, llm, embedding))

    # Method 2: vocabulary (near-zero LLM cost)
    results.append(run_method("vocabulary", doc_text, llm, embedding))

    # Method 3: llm (full LLM cost)
    results.append(run_method("llm", doc_text, llm, embedding))

    # Summary
    print(f"\n\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Method':<15} {'KUs':>6} {'Entities':>10} {'Edges':>8} {'Index(s)':>10} {'TopScore':>10}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['method']:<15} {r['kus']:>6} {r['entities']:>10} {r['edges']:>8} {r['index_time']:>9.1f}s {r['top_score']:>9.3f}"
        )
    print(f"{'=' * 60}")
    print("\nALL METHODS TESTED SUCCESSFULLY")


if __name__ == "__main__":
    main()
