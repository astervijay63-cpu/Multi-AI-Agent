from functools import lru_cache
from graph.hitl import HITLGate
from memory.episodic_store import EpisodicStore
from memory.procedural_store import ProceduralStore
from memory.semantic_store import SemanticStore


@lru_cache(maxsize=1)
def get_hitl_gate() -> HITLGate:
    """
    Returns the shared HITLGate instance.

    maxsize=1 means: cache the result of the first call.
    Every subsequent call returns the same HITLGate object.
    The HITLGate holds a database connection pool — we want
    exactly ONE pool, not a new one per request.

    Returns:
        The singleton HITLGate instance.
    """
    return HITLGate()


@lru_cache(maxsize=1)
def get_episodic_store() -> EpisodicStore:
    """
    Returns the shared EpisodicStore instance.
    One instance = one connection pool = efficient resource use.
    """
    return EpisodicStore()


@lru_cache(maxsize=1)
def get_procedural_store() -> ProceduralStore:
    """
    Returns the shared ProceduralStore instance.
    """
    return ProceduralStore()


@lru_cache(maxsize=1)
def get_semantic_store() -> SemanticStore:
    """
    Returns the shared SemanticStore instance.
    """
    return SemanticStore()


def get_graph():
    """
    Returns the compiled MOSAIC LangGraph graph.

    NOT cached with lru_cache because mosaic_graph is already
    a module-level singleton — it was compiled once in graph_builder.py
    and importing it here gives us the same object every time.

    Returns:
        The compiled LangGraph graph ready to invoke.
    """
    # Agent modules create their Gemini clients at import time.  Defer that
    # work until an analysis is requested so localhost can serve the UI and
    # docs before integration credentials are configured.
    from graph.graph_builder import mosaic_graph

    return mosaic_graph
