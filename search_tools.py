import asyncio
import json
from langchain_core.tools import tool
from processing.vector_store import VectorStore
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from processing.embedder import Embedder
from config.logging_config import setup_logging

logger = setup_logging(__name__)

_embedder       = Embedder()
_vector_store   = VectorStore()
_episodic_store = EpisodicStore()
_semantic_store = SemanticStore()

# Guards _ensure_vector_store below. The six specialist agents run in
# parallel, so several tools reach that function at the same instant.
_vector_store_lock = asyncio.Lock()


async def _ensure_vector_store() -> None:
    """
    Lazily connects the shared VectorStore singleton.

    Unlike EpisodicStore/SemanticStore, VectorStore does not
    self-heal its connection pool on every call — it requires an
    explicit init() (see processing/vector_store.py). Since this
    module keeps one VectorStore instance alive for the whole
    process, we only need to do that once.

    WHY THE LOCK:
    A bare "if not initialised: await init()" is check-then-act, and
    the await inside init() yields control. With six agents running
    concurrently, four tool calls all passed the check before any of
    them finished connecting — Postgres saw four pools created, and
    three of them were orphaned by the final assignment to _pool:
    unreachable, never closed, holding connections open against Neon.
    The lock makes the first caller connect while the rest wait, and
    the second check inside it stops them re-connecting afterwards.
    """
    if _vector_store.is_initialised:
        return

    async with _vector_store_lock:
        if not _vector_store.is_initialised:
            await _vector_store.init()


@tool
async def search_studies_by_meaning(
    query: str,
    top_k: int = 5,
    source_filter: str = "study",
) -> str:
    """
    Search clinical trial studies using semantic similarity.

    Use this tool when you need to find studies related to a specific
    topic, condition, sponsor behaviour, or research integrity issue.
    The search works by MEANING — not exact keyword matching.

    For example:
    - "studies where sponsor never posted results" finds studies about
      missing results even if they use different words
    - "Novo Nordisk cardiovascular trials" finds all relevant chunks

    Args:
        query:         What to search for. Write as a natural language question.
        top_k:         How many results to return. Default 5. Max 10.
        source_filter: "study" to search trial records only.
                       "paper" to search PubMed papers only.
                       Leave as "study" for most agent tasks.

    Returns:
        JSON string containing matching study chunks with similarity scores.
    """

    logger.info(
        f"Tool called: search_studies_by_meaning | "
        f"query='{query[:60]}' | top_k={top_k}"
    )
 
    try:
        await _ensure_vector_store()
        query_embedding = await _embedder.embed_text(query)
        results = await _vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            source_filter=source_filter,
        )

        if not results:
            return json.dumps({
                "results": [],
                "message": "No relevant studies found for this query.",
                "query":   query,
            })

        return json.dumps({
            "results": results,
            "count":   len(results),
            "query":   query,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"search_studies_by_meaning failed | error={e}")
        return json.dumps({"error": str(e), "results": []})


@tool
async def search_past_episodes(
    query: str,
    agent_name: str,
    top_k: int = 3,
) -> str:
    """
    Search through past agent reasoning sessions (episodic memory).

    Use this tool at the START of every investigation to check if
    you have found similar signals before. This prevents duplicate
    work and gives you historical context.

    Ask questions like:
    - "previous findings about missing results from this sponsor"
    - "past investigations of NCT04788680"
    - "episodes where outcome switching was detected"

    Args:
        query:      What to search for in past episodes.
        agent_name: Your own agent name — filters to YOUR past sessions.
                    Example: "missing_results_agent"
        top_k:      How many past episodes to retrieve. Default 3.

    Returns:
        JSON string with the most relevant past episodes.
        If empty, this is the first time investigating this topic.
    """

    logger.info(
        f"Tool called: search_past_episodes | "
        f"agent={agent_name} | query='{query[:60]}'"
    )

    try:
        episodes = await _episodic_store.search_episodes(
            query=query,
            agent_name=agent_name,
            top_k=top_k,
        )

        if not episodes:
            return json.dumps({
                "episodes": [],
                "message":  "No relevant past episodes found. "
                            "This appears to be a new type of investigation.",
                "query":    query,
            })

        return json.dumps({
            "episodes": episodes,
            "count":    len(episodes),
            "query":    query,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"search_past_episodes failed | error={e}")
        return json.dumps({"error": str(e), "episodes": []})


@tool
async def save_episode(
    agent_name: str,
    content: str,
    nct_id: str = "",
    outcome: str = "completed",
) -> str:
    """
    Save the current reasoning session as an episode in memory.

    Call this tool at the END of every investigation — after you have
    drawn your conclusions. Saving episodes builds your long-term memory
    so future sessions can benefit from what you found today.

    Write the content as a detailed case note:
    - What study you investigated
    - What the sponsor's behaviour was
    - What signals you found or did not find
    - Why you reached your conclusion

    Args:
        agent_name: Your own agent name.
                    Example: "missing_results_agent"
        content:    Detailed description of what you investigated and found.
                    Write this like a detective's case note.
        nct_id:     The NCT ID of the study you investigated.
                    Leave empty if investigating multiple studies.
        outcome:    What happened: "signal_generated", "no_signal",
                    "sent_to_review", or "completed".

    Returns:
        JSON string confirming the episode was saved with its ID.
    """

    logger.info(
        f"Tool called: save_episode | "
        f"agent={agent_name} | nct_id={nct_id} | outcome={outcome}"
    )

    try:
        episode_id = await _episodic_store.save_episode(
            agent_name=agent_name,
            content=content,
            nct_id=nct_id if nct_id else None,
            outcome=outcome,
        )

        return json.dumps({
            "success":    True,
            "episode_id": episode_id,
            "message":    "Episode saved to long-term memory successfully.",
            "agent":      agent_name,
        }, indent=2)

    except Exception as e:
        logger.error(f"save_episode failed | error={e}")
        return json.dumps({"success": False, "error": str(e)})


@tool
async def get_sponsor_profile(sponsor_name: str) -> str:
    """
    Retrieve everything MOSAIC knows about a specific research sponsor.

    Use this tool when evaluating a study to understand the sponsor's
    historical behaviour — their compliance record, broken promises,
    average delays, and credibility score.

    A credibility score below 0.6 is concerning.
    A credibility score below 0.4 is a serious red flag.

    Args:
        sponsor_name: The exact sponsor name as it appears in the study.
                      Example: "Novo Nordisk A/S"
                      Example: "National Cancer Institute (NCI)"

    Returns:
        JSON string with the sponsor's full profile.
        If the sponsor is new (never analysed before), returns a message
        indicating no historical data is available.
    """

    logger.info(
        f"Tool called: get_sponsor_profile | sponsor={sponsor_name}"
    )

    try:
        profile = await _semantic_store.get_sponsor_profile(sponsor=sponsor_name)

        if profile is None:
            return json.dumps({
                "sponsor":  sponsor_name,
                "found":    False,
                "message":  f"No historical data for '{sponsor_name}'. "
                            "This sponsor has not been analysed before. "
                            "Proceed with lower confidence.",
            }, indent=2)

        return json.dumps({
            "found":   True,
            "profile": profile,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"get_sponsor_profile failed | error={e}")
        return json.dumps({"error": str(e), "found": False})


@tool
async def update_sponsor_profile(
    sponsor_name:       str,
    results_posted:     bool = False,
    had_broken_promise: bool = False,
    delay_days:         int  = 0,
) -> str:
    """
    Update a sponsor's profile with findings from the current study.

    Call this tool AFTER you have analysed a study and determined:
    - Whether the sponsor posted results (True/False)
    - Whether outcome switching was detected (True/False)
    - How many days late the study was (0 if on time)

    This update accumulates in the sponsor's profile permanently.
    Future agent sessions will see the updated credibility score.

    Args:
        sponsor_name:       The exact sponsor name from the study.
        results_posted:     True if sponsor posted results, False if not.
        had_broken_promise: True if outcome switching was detected.
        delay_days:         How many days past completion date. 0 if on time.

    Returns:
        JSON string confirming the update was applied.
    """

    logger.info(
        f"Tool called: update_sponsor_profile | "
        f"sponsor={sponsor_name} | "
        f"results_posted={results_posted} | "
        f"broken_promise={had_broken_promise} | "
        f"delay_days={delay_days}"
    )

    try:
        await _semantic_store.update_sponsor_knowledge(
            sponsor=sponsor_name,
            results_posted=results_posted,
            had_broken_promise=had_broken_promise,
            delay_days=delay_days,
        )

        return json.dumps({
            "success":      True,
            "sponsor":      sponsor_name,
            "message":      "Sponsor profile updated successfully.",
            "results_posted":     results_posted,
            "had_broken_promise": had_broken_promise,
            "delay_days":         delay_days,
        }, indent=2)

    except Exception as e:
        logger.error(f"update_sponsor_profile failed | error={e}")
        return json.dumps({"success": False, "error": str(e)})


@tool
async def get_low_credibility_sponsors(
    threshold:   float = 0.6,
    min_studies: int   = 3,
) -> str:
    """
    Get all sponsors with credibility scores below the threshold.

    Use this tool when looking for patterns across problematic sponsors
    or when you want to check if the current study's sponsor has a
    history of compliance issues.

    Args:
        threshold:   Credibility below this is considered low. Default 0.6.
        min_studies: Minimum studies to qualify. Avoids judging new sponsors
                     on too little data. Default 3.

    Returns:
        JSON string listing all low-credibility sponsors with their profiles.
        Empty list if all sponsors are above the threshold.
    """

    logger.info(
        f"Tool called: get_low_credibility_sponsors | "
        f"threshold={threshold} | min_studies={min_studies}"
    )

    try:
        sponsors = await _semantic_store.get_low_credibility_sponsors(
            threshold=threshold,
            min_studies=min_studies,
        )

        if not sponsors:
            return json.dumps({
                "sponsors": [],
                "message":  f"No sponsors found below credibility {threshold} "
                            f"with at least {min_studies} studies.",
                "count":    0,
            }, indent=2)

        return json.dumps({
            "sponsors": sponsors,
            "count":    len(sponsors),
            "threshold": threshold,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"get_low_credibility_sponsors failed | error={e}")
        return json.dumps({"error": str(e), "sponsors": []})


@tool
async def search_study_chunks_by_nct_id(
    nct_id: str,
    query:  str = "",
) -> str:
    """
    Retrieve all text chunks for one specific study by its NCT ID.

    Use this tool when you already know WHICH study you want to
    examine in detail and need to read its full content.

    Different from search_studies_by_meaning which searches ACROSS
    all studies. This tool gets the full content of ONE specific study.

    Args:
        nct_id: The specific study's NCT ID.
                Example: "NCT04788680"
        query:  Optional — if provided, returns only the most relevant
                chunk for this study. Leave empty to get all chunks.

    Returns:
        JSON string with all chunks from this study.
    """

    logger.info(
        f"Tool called: search_study_chunks_by_nct_id | "
        f"nct_id={nct_id}"
    )

    try:
        await _ensure_vector_store()
        if query:
            query_embedding = await _embedder.embed_text(query)
            results = await _vector_store.search(
                query_embedding=query_embedding,
                top_k=5,
                nct_id_filter=nct_id,
            )
        else:
            results = await _vector_store.get_chunks_for_study(nct_id=nct_id)

        if not results:
            return json.dumps({
                "nct_id":  nct_id,
                "chunks":  [],
                "message": f"No chunks found for study {nct_id}. "
                           "The study may not have been processed yet.",
            })

        return json.dumps({
            "nct_id": nct_id,
            "chunks": results,
            "count":  len(results),
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"search_study_chunks_by_nct_id failed | "
            f"nct_id={nct_id} | error={e}"
        )
        return json.dumps({"error": str(e), "chunks": []})


@tool
async def search_papers_by_meaning(
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search PubMed research papers using semantic similarity.

    Use this tool when you need to find published research papers
    related to a specific topic, drug, or safety concern.

    Different from search_studies_by_meaning which searches clinical
    trial FILINGS. This searches published RESEARCH PAPERS.

    The Side Effect Checker agent uses this most heavily — comparing
    what official filings say against what papers reported.

    Args:
        query: What to search for in published papers.
               Example: "semaglutide cardiovascular side effects"
               Example: "NCT04788680 safety outcomes"
        top_k: How many results to return. Default 5.

    Returns:
        JSON string with matching paper chunks and similarity scores.
    """

    logger.info(
        f"Tool called: search_papers_by_meaning | "
        f"query='{query[:60]}' | top_k={top_k}"
    )

    try:
        await _ensure_vector_store()
        query_embedding = await _embedder.embed_text(query)
        results = await _vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            source_filter="paper",
        )

        if not results:
            return json.dumps({
                "results": [],
                "message": "No relevant papers found for this query.",
                "query":   query,
            })

        return json.dumps({
            "results": results,
            "count":   len(results),
            "query":   query,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"search_papers_by_meaning failed | error={e}")
        return json.dumps({"error": str(e), "results": []})


ALL_SEARCH_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
    get_low_credibility_sponsors,
    search_study_chunks_by_nct_id,
    search_papers_by_meaning,
]