import json
from langchain_core.tools import tool
from ingestion.pubmed_client import PubMedClient
from ingestion.document_parser import DocumentParser
from config.logging_config import setup_logging

logger = setup_logging(__name__)

_parser = DocumentParser()


@tool
async def fetch_papers_for_trial(
    nct_id: str,
    max_papers: int = 10,
) -> str:
    """
    Fetch all published research papers that reference a specific
    clinical trial — LIVE from PubMed right now.

    This is the primary tool for the Side Effect Checker agent.
    It finds papers where authors reported their findings about
    a specific trial — then the agent compares those findings
    against what the official trial filing says.

    WHY THIS IS POWERFUL:
    Official filings are written by the sponsor.
    Published papers are written by independent researchers.
    When these two sources DISAGREE about safety or outcomes,
    that disagreement is a signal worth investigating.

    Example scenario:
    Official filing says: "No serious adverse events observed"
    Published paper says: "Three patients were hospitalised"
    → Side Effect Checker flags this as a safety gap signal.

    Args:
        nct_id:     The trial's NCT ID to search for in PubMed.
                    Example: "NCT04788680"
        max_papers: Maximum papers to fetch. Default 10.
                    Keep low — each paper adds to agent context window.
                    More than 15 papers can overwhelm the agent.

    Returns:
        JSON string with all papers found, including:
        - title, abstract, journal, authors, publication date
        - word_count (very short abstracts may indicate limited detail)
        Empty list if no papers reference this trial in PubMed.
    """

    logger.info(
        f"Tool called: fetch_papers_for_trial | "
        f"nct_id={nct_id} | max_papers={max_papers}"
    )

    try:
        async with PubMedClient() as client:
            raw_papers = await client.fetch_papers_for_trial(
                nct_id=nct_id,
                max_results=max_papers,
            )

        if not raw_papers:
            return json.dumps({
                "nct_id":  nct_id,
                "papers":  [],
                "count":   0,
                "message": f"No published papers found on PubMed that "
                           f"reference trial {nct_id}. The trial may not "
                           "have published results in academic journals, "
                           "or results may only exist as grey literature.",
            }, indent=2)

        parsed_papers = _parser.parse_papers(raw_papers=raw_papers)

        papers_list = []
        for paper in parsed_papers:

            paper_dict = paper.model_dump()

            papers_list.append({
                "pmid":               paper_dict["pmid"],
                "title":              paper_dict["title"],
                "abstract":           paper_dict["abstract"],
                "journal":            paper_dict["journal"],
                "pub_date":           paper_dict["pub_date"],
                "authors":            paper_dict["authors"][:5],
                "word_count":         paper_dict["word_count"],
                "nct_ids_referenced": paper_dict["nct_ids_referenced"],
            })

        return json.dumps({
            "nct_id": nct_id,
            "papers": papers_list,
            "count":  len(papers_list),
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"fetch_papers_for_trial failed | "
            f"nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error":  str(e),
            "papers": [],
            "count":  0,
        })


@tool
async def search_pubmed_by_query(
    query: str,
    max_papers: int = 5,
) -> str:
    """
    Search PubMed with a free-text query and fetch matching papers.

    Use this tool when you want to find papers about a topic,
    drug, or condition — not just papers about one specific trial.

    Different from fetch_papers_for_trial which searches by NCT ID.
    This tool accepts any PubMed search query.

    Examples:
    - "semaglutide cardiovascular outcomes 2023"
    - "metformin diabetes safety adverse events"
    - "Novo Nordisk clinical trial results transparency"

    The Pattern Finder agent uses this to find papers that discuss
    multiple trials from the same sponsor — revealing systemic patterns.

    Args:
        query:      Any PubMed-compatible search query.
                    Can include drug names, conditions, author names,
                    journal names, or any combination.
        max_papers: Maximum papers to return. Default 5.
                    Keep low — each paper adds to agent context.

    Returns:
        JSON string with matching papers from PubMed.
    """

    logger.info(
        f"Tool called: search_pubmed_by_query | "
        f"query='{query[:60]}' | max_papers={max_papers}"
    )

    try:
        async with PubMedClient() as client:
            paper_ids = await client._search_paper_ids(
                nct_id=query,
                max_results=max_papers,
            )

            raw_papers = (
                await client._fetch_paper_details(paper_ids=paper_ids)
                if paper_ids else []
            )

        if not raw_papers:
            return json.dumps({
                "query":   query,
                "papers":  [],
                "count":   0,
                "message": f"No papers found on PubMed for query: '{query}'. "
                           "Try a broader search term or different keywords.",
            }, indent=2)

        parsed_papers = _parser.parse_papers(raw_papers=raw_papers)

        papers_list = []
        for paper in parsed_papers:
            paper_dict = paper.model_dump()

            papers_list.append({
                "pmid":               paper_dict["pmid"],
                "title":              paper_dict["title"],
                "abstract":           paper_dict["abstract"],
                "journal":            paper_dict["journal"],
                "pub_date":           paper_dict["pub_date"],
                "authors":            paper_dict["authors"][:5],
                "word_count":         paper_dict["word_count"],
                "nct_ids_referenced": paper_dict["nct_ids_referenced"],
            })

        return json.dumps({
            "query":  query,
            "papers": papers_list,
            "count":  len(papers_list),
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"search_pubmed_by_query failed | "
            f"query={query} | error={e}"
        )
        return json.dumps({
            "query":  query,
            "error":  str(e),
            "papers": [],
            "count":  0,
        })


@tool
async def compare_filing_vs_papers(
    nct_id: str,
    filing_summary: str,
) -> str:
    """
    Fetch papers for a trial and compare them against the official filing.

    This is the Side Effect Checker's most powerful tool.
    It fetches all published papers about a trial, then returns
    both the official filing summary AND the papers — so the agent
    can compare them and identify discrepancies.

    The agent looks for:
    - Safety events mentioned in papers but not in the filing
    - Different severity levels (filing says "mild", paper says "serious")
    - Outcomes reported in papers that differ from the primary outcome
    - Results published in papers when no official results were posted

    Args:
        nct_id:          The trial's NCT ID.
        filing_summary:  A summary of what the official filing says.
                         The agent provides this from earlier tool calls.
                         Example: "Filing reports no serious adverse events.
                                   Primary outcome was HbA1c reduction.
                                   Results posted: No."

    Returns:
        JSON string with:
        - filing_summary: what the agent passed in (echoed back)
        - papers: all published papers found on PubMed
        - comparison_note: guidance on what to look for
        - papers_count: how many papers were found
    """

    logger.info(
        f"Tool called: compare_filing_vs_papers | nct_id={nct_id}"
    )

    try:
        async with PubMedClient() as client:
            raw_papers = await client.fetch_papers_for_trial(
                nct_id=nct_id,
                max_results=15,
            )
        parsed_papers = _parser.parse_papers(raw_papers=raw_papers or [])

        papers_list = []
        for paper in parsed_papers:
            paper_dict = paper.model_dump()
            papers_list.append({
                "pmid":     paper_dict["pmid"],
                "title":    paper_dict["title"],
                "abstract": paper_dict["abstract"],
                "journal":  paper_dict["journal"],
                "pub_date": paper_dict["pub_date"],
                "authors":  paper_dict["authors"][:3],
            })

        comparison_note = (
            "Compare the filing_summary above against each paper's abstract. "
            "Look specifically for: "
            "(1) adverse events mentioned in papers but absent from filing, "
            "(2) different severity descriptions for the same event, "
            "(3) outcome results that contradict the filing's claims, "
            "(4) results data in papers when filing shows results_posted=False."
        )

        return json.dumps({
            "nct_id":          nct_id,
            "filing_summary":  filing_summary,
            "papers":          papers_list,
            "papers_count":    len(papers_list),
            "comparison_note": comparison_note,
            "has_papers": len(papers_list) > 0,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"compare_filing_vs_papers failed | "
            f"nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error":  str(e),
            "papers": [],
        })


ALL_PUBMED_TOOLS = [
    fetch_papers_for_trial,
    search_pubmed_by_query,
    compare_filing_vs_papers,
]