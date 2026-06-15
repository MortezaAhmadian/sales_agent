"""LangChain tools exposed to the agent for RAG query and memory writing.

Set the active store with set_store() before the agent loop starts.
"""

from langchain_core.tools import tool

from rag_store import DocChunk, RAGStore

_store: RAGStore | None = None


def set_store(store: RAGStore) -> None:
    global _store
    _store = store


@tool
def query_knowledge_base(
    query: str,
    component: str = "",
    doc_type: str = "",
    k: int = 5,
) -> str:
    """Search the RAG knowledge base for deployment procedures, configs, and troubleshooting.

    CALL THIS FIRST before every deployment step to retrieve the exact commands and configs.
    The knowledge base contains: Helm chart values, MOP procedures, Linux networking commands,
    Kubernetes operations, Multus CNI setup, troubleshooting patterns, and cluster-specific fixes.

    Args:
        query:     What you need (e.g. "helm install SMF PLMN 999/70", "UPF iptables dataplane fix",
                   "Calico restart stale token worker2", "UERANSIM SCTP gNB values")
        component: Optional filter — amf | smf | upf | nrf | udm | udr | ausf | nssf | pcf |
                   ueransim | mongodb | calico | kubernetes | linux | general
        doc_type:  Optional filter — procedure | config | documentation | troubleshooting | reference
        k:         Number of results to return (default 5)

    Returns:
        Formatted string of retrieved chunks with source, component, and relevance score.
        Memory entries (from past deployment runs) are tagged [MEMORY].
    """
    if _store is None:
        return "ERROR: RAG store not initialised. Run ingest.py first."

    results = _store.query(
        query_text=query,
        k=k,
        component=component or None,
        doc_type=doc_type or None,
    )
    if not results:
        return f"No results found for: '{query}'"

    lines = [f"=== RAG: '{query}' ({len(results)} results) ===\n"]
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        src  = meta.get("source_file", meta.get("source", "?"))
        comp = meta.get("component", "")
        mem  = " [MEMORY]" if meta.get("from_memory") else ""
        lines.append(f"[{i}] {src}  component={comp}  score={r['score']}{mem}")
        lines.append(r["content"])
        lines.append("")
    return "\n".join(lines)


@tool
def save_deployment_memory(
    fact: str,
    component: str = "general",
    phase: str = "",
    tags: str = "",
) -> dict:
    """Save a learned fact, working fix, or error pattern to deployment memory.

    Call this whenever:
    - A command or config sequence successfully resolved an issue
    - You discover a cluster-specific behaviour worth remembering
    - An error message maps to a specific root cause + fix

    Args:
        fact:      The fact or fix in plain text (include the exact commands if relevant)
        component: NF or subsystem: amf | smf | upf | nrf | calico | kubernetes | linux | general
        phase:     Deployment phase: setup | deploy | dataplane | verify | troubleshoot
        tags:      Comma-separated keywords for future retrieval (e.g. "iptables,nat,masquerade")

    Returns:
        {"success": bool, "saved": str (first 80 chars of fact)}
    """
    if _store is None:
        return {"success": False, "error": "RAG store not initialised"}

    chunk = DocChunk(
        content=fact,
        metadata={
            "source":    "agent_memory",
            "doc_type":  "memory",
            "component": component,
            "phase":     phase,
            "tags":      tags,
        },
    )
    _store.upsert_chunks([chunk], collection="memory")
    preview = (fact[:77] + "...") if len(fact) > 80 else fact
    return {"success": True, "saved": preview}
