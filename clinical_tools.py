import json
from langchain_core.tools import tool
from ingestion.clinical_trials_client import ClinicalTrialsClient
from ingestion.document_parser import DocumentParser
from config.logging_config import setup_logging
from tools.study_data_tools import search_local_study_data

logger = setup_logging(__name__)


_parser = DocumentParser()


@tool
async def fetch_study_details(nct_id: str) -> str:
    """
    Fetch the complete, LIVE record for one specific clinical trial
    directly from ClinicalTrials.gov API.

    Use this tool when:
    - You need the most current version of a study (freshest data)
    - You need fields that may not be in our local database
    - The study was not in our original ingestion corpus

    This makes a LIVE API call — slightly slower than database queries.
    Prefer database search tools when current data is not critical.

    Args:
        nct_id: The study's unique identifier.
                Format: "NCT" followed by 8 digits.
                Example: "NCT04788680"

    Returns:
        JSON string with the complete cleaned study record.
        Returns an error message if the study is not found.
    """

    logger.info(
        f"Tool called: fetch_study_details | nct_id={nct_id}"
    )

    try:
        async with ClinicalTrialsClient() as client:
            raw_study = await client.fetch_study(nct_id=nct_id)

        if raw_study is None:
            return json.dumps({
                "found":   False,
                "nct_id":  nct_id,
                "message": f"Study {nct_id} was not found on "
                           "ClinicalTrials.gov. The NCT ID may be "
                           "incorrect or the study may have been removed.",
            }, indent=2)

        parsed_study = _parser.parse_study(raw=raw_study)

        if parsed_study is None:
            return json.dumps({
                "found":   False,
                "nct_id":  nct_id,
                "message": "Study was found but could not be parsed. "
                           "The API response had an unexpected structure.",
            }, indent=2)

        study_dict = parsed_study.model_dump()

        study_dict.pop("raw_data", None)

        return json.dumps({
            "found":   True,
            "nct_id":  nct_id,
            "study":   study_dict,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"fetch_study_details failed | nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "found": False,
            "error": str(e),
            "nct_id": nct_id,
        })


@tool
async def search_studies_by_condition(
    condition: str,
    max_results: int = 10,
    status_filter: str = "COMPLETED",
) -> str:
    """
    Search ClinicalTrials.gov LIVE for studies matching a condition.

    Use this tool when you want to find trials for a specific
    medical condition that may not be in our local database.

    Examples of when to use this:
    - "Find all completed diabetes trials from the last 5 years"
    - "Search for cardiovascular trials from Pfizer"
    - "Find recruiting trials for this drug"

    This makes LIVE API calls. Results may differ from our database
    because ClinicalTrials.gov updates continuously.

    Args:
        condition:     Medical condition to search for.
                       Example: "diabetes", "cancer", "heart failure"
        max_results:   Maximum studies to return. Default 10. Max 50.
                       Keep low to avoid slow tool calls.
        status_filter: Filter by study status.
                       "COMPLETED"            → only completed studies
                       "RECRUITING"           → only recruiting studies
                       "ACTIVE_NOT_RECRUITING"→ ongoing but not recruiting
                       Default "COMPLETED" — most relevant for signal detection.

    Returns:
        JSON string with a list of matching studies (cleaned format).
    """

    logger.info(
        f"Tool called: search_studies_by_condition | "
        f"condition={condition} | max_results={max_results}"
    )

    try:
        async with ClinicalTrialsClient() as client:
            raw_studies = await client.search_studies(
                condition=condition,
                status=[status_filter] if status_filter else None,
                max_results=min(max_results, 50),
            )

        if not raw_studies:
            return json.dumps({
                "studies": [],
                "count":   0,
                "message": f"No studies found for condition '{condition}' "
                           f"with status '{status_filter}'.",
            }, indent=2)

        parsed_studies = _parser.parse_studies(raw_studies=raw_studies)

        studies_list = []
        for study in parsed_studies:
            study_dict = study.model_dump()

            study_dict.pop("raw_data", None)

            study_dict.pop("protocol_amendments", None)

            studies_list.append(study_dict)

        return json.dumps({
            "studies":      studies_list,
            "count":        len(studies_list),
            "condition":    condition,
            "status_filter": status_filter,
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"search_studies_by_condition failed | "
            f"condition={condition} | error={e}"
        )
        return json.dumps({
            "error":   str(e),
            "studies": [],
            "count":   0,
        })


@tool
async def check_results_posted(nct_id: str) -> str:
    """
    Check if a specific clinical trial has posted results — RIGHT NOW.

    Use this tool when you need the CURRENT results posting status
    for a study. This gives you the live status from ClinicalTrials.gov,
    not the status stored during ingestion (which may be outdated).

    This is the most important tool for the Missing Results Agent.
    A study that had no results during ingestion may have posted
    results since then — this tool catches that.

    Args:
        nct_id: The study's NCT ID to check.
                Example: "NCT04788680"

    Returns:
        JSON string with:
        - results_posted: True if results are posted, False if not
        - completion_date: When the study completed
        - status: Current study status
        - months_overdue: How many months past the 12-month deadline
                          (only present if results are missing)
    """

    logger.info(
        f"Tool called: check_results_posted | nct_id={nct_id}"
    )

    try:
        async with ClinicalTrialsClient() as client:
            raw_study = await client.fetch_study(nct_id=nct_id)

        if raw_study is None:
            return json.dumps({
                "nct_id":  nct_id,
                "found":   False,
                "message": f"Study {nct_id} not found on ClinicalTrials.gov.",
            }, indent=2)

        parsed = _parser.parse_study(raw=raw_study)

        if parsed is None:
            return json.dumps({
                "nct_id":  nct_id,
                "found":   False,
                "message": "Could not parse study response.",
            }, indent=2)

        result = {
            "nct_id":          parsed.nct_id,
            "found":           True,
            "results_posted":  parsed.results_posted,
            "status":          parsed.status,
            "completion_date": parsed.completion_date,
            "sponsor":         parsed.sponsor,
        }

        if not parsed.results_posted and parsed.status == "COMPLETED":

            if parsed.completion_date:
                try:
                    from datetime import datetime

                    completion = datetime.strptime(
                        parsed.completion_date[:7],
                        "%Y-%m"
                    )

                    now = datetime.utcnow()

                    months_since_completion = (
                        (now.year - completion.year) * 12
                        + (now.month - completion.month)
                    )

                    months_overdue = months_since_completion - 12

                    if months_overdue > 0:
                        result["months_overdue"] = months_overdue
                        result["years_overdue"]  = round(
                            months_overdue / 12, 1
                        )

                        result["is_violation"] = True

                except ValueError:
                    pass

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"check_results_posted failed | nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error":  str(e),
            "found":  False,
        })


@tool
async def get_study_amendments(nct_id: str) -> str:
    """
    Fetch the protocol amendment history for a specific clinical trial.

    Use this tool when investigating whether a study changed its
    primary outcomes or design mid-study — the core signal for
    the Broken Promises Agent.

    Protocol amendments are official, time-stamped changes to the
    study design filed with ClinicalTrials.gov. They tell us:
    - When the study design changed
    - What changed
    - Whether the change happened BEFORE or AFTER enrollment began

    Timing matters: changes AFTER enrollment has started are more
    suspicious than changes before enrollment began.

    Args:
        nct_id: The study's NCT ID.
                Example: "NCT04788680"

    Returns:
        JSON string with the amendment history and key timing details.
    """

    logger.info(
        f"Tool called: get_study_amendments | nct_id={nct_id}"
    )

    try:
        async with ClinicalTrialsClient() as client:
            raw_study = await client.fetch_study(nct_id=nct_id)

        if raw_study is None:
            return json.dumps({
                "nct_id":     nct_id,
                "found":      False,
                "amendments": [],
            }, indent=2)

        parsed = _parser.parse_study(raw=raw_study)

        if parsed is None:
            return json.dumps({
                "nct_id":     nct_id,
                "found":      False,
                "amendments": [],
                "message":    "Could not parse study.",
            }, indent=2)

        return json.dumps({
            "nct_id":          parsed.nct_id,
            "found":           True,
            "title":           parsed.title,
            "sponsor":         parsed.sponsor,
            "start_date":      parsed.start_date,
            "completion_date": parsed.completion_date,
            "primary_outcome": parsed.primary_outcome,
            "amendments":      parsed.protocol_amendments,
            "amendment_count": len(parsed.protocol_amendments),
        }, indent=2, default=str)

    except Exception as e:
        logger.error(
            f"get_study_amendments failed | nct_id={nct_id} | error={e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error":  str(e),
            "found":  False,
        })


ALL_CLINICAL_TOOLS = [
    fetch_study_details,
    search_studies_by_condition,
    check_results_posted,
    get_study_amendments,
    search_local_study_data,
]