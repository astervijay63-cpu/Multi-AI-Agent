# Fetch All the Clinical Study records from clinicaltrials.gov
import asyncio
import requests
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.logging_config import setup_logging
from config.settings import settings

logger = setup_logging(__name__)

BASE_URL = settings.clinical_trials_base_url
PAGE_SIZE = settings.clinical_trials_page_size
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
HEADERS = {
    "Accept":"application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

class ClinicalTrialsClient:
    """
    This client handles everything needed to talk to API
    """
    def __init__(self):
        self._sessions:requests.Session | None  = None

    async def __aenter__(self) -> "ClinicalTrialsClient":
        """
        called when we enter async block
        created the HTTP session and sets the shared headers
        """
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        logger.info("ClinicalTrials client opened")
        return self

    async def __aexit__(
            self, 
            exc_type, 
            exc_val, exc_tb
    ):
        """
        this is called when we exit async block
        closes the HTTP session and releases the connection
        runs even on error in async block
        """
        if self._session:
            # close only when session is actually created
            self._session.close()
            logger.info("Clinical Trials Client closed")
        
    async def search_studies(
            self, 
            condition:str | None = None,
            intervention:str | None = None,
            sponsor:str | None = None,
            status: list[str] | None = None,
            max_results: int = 100
    )-> list[dict[str,Any]]:
        """
        Searches ClinicalTrials.gov and returns a list of study records.

        This is the main method you call to get study data.
        It handles everything internally:
        - Building the search parameters
        - Fetching multiple pages until max_results is reached
        - Returning all results as a flat list of dictionaries

        Args:
            condition:   Medical condition to search for.
            intervention: Drug or treatment to search for.
            sponsor:     Organisation running the study.
            status:      List of study statuses to filter by.
            max_results: Maximum total studies to return.

        Returns:
            List of raw study dictionaries exactly as the API returned them.
            Each dictionary contains all the study fields —
            nct_id, title, sponsor, outcomes, dates, etc.
        """
        all_studies: list[dict[str,Any]] = [] 
        next_page_token:str | None = None
        page_number = 0
        logger.info(
            f"Searching studies | "
            f"condition={condition} | "
            f"intervention={intervention} | "
            f"sponsor={sponsor} | "
            f"max_results={max_results}"
        )

        # pagination loop
        while len(all_studies) < max_results:
            page_number  = page_number + 1

            params = self._build_search_params(
                condition = condition,
                intervention = intervention,
                sponsor = sponsor,
                status = status,
                page_token = next_page_token
            )

            response_data = await self._fetch_page(params=params)

            if not response_data:
                break

            page_studies = response_data.get('studies',[])

            if not page_studies:
                logger.info('No more studies available - pagination complete')
                break
            all_studies.extend(page_studies)

            logger.info(
                f"Page {page_number} | "
                f"fetched={len(page_studies)} | "
                f"total so far={len(all_studies)}"
            )

            next_page_token = response_data.get("nextPageToken")

            if not next_page_token:
                logger.info("Last page reached — no nextPageToken in response")
                break

        all_studies = all_studies[:max_results]

        logger.info(
            f"Search complete | "
            f"total studies returned={len(all_studies)}"
        )

        return all_studies


    async def fetch_study(self, nct_id: str) -> dict[str, Any] | None:
        """
        Fetches the complete record for one specific study by its NCT ID.

        NCT ID is the unique identifier every study gets when it registers
        on ClinicalTrials.gov. Format: NCT followed by 8 digits.
        Example: NCT04788680

        Use this when you already know which specific study you want
        and need its full details — not for searching.

        Args:
            nct_id: The study's unique identifier. Example: "NCT04788680"

        Returns:
            A dictionary with all the study's details.
            None if the study was not found or the request failed.
        """

        logger.info(f"Fetching single study | nct_id={nct_id}")

        def _get_study():
            return self._session.get(
                f"{BASE_URL}/studies/{nct_id}",
                timeout=REQUEST_TIMEOUT,
            )

        try:
            response = await asyncio.to_thread(_get_study)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"Study not found | "
                f"nct_id={nct_id} | "
                f"status={e.response.status_code}"
            )
            return None

        except Exception as e:
            logger.error(
                f"Failed to fetch study | "
                f"nct_id={nct_id} | "
                f"error={e}"
            )
            return None

    def _build_search_params(
        self,
        condition: str | None,
        intervention: str | None,
        sponsor: str | None,
        status: list[str] | None,
        page_token: str | None,
    ) -> dict[str, Any]:
        """
        Builds the query parameter dictionary for one API request.

        The ClinicalTrials.gov API expects specific parameter names.
        This method translates our friendly Python arguments into
        the exact parameter names the API understands.

        Only includes parameters that were actually provided —
        if condition is None, we do not add query.cond to the request.

        Args:
            condition:    Medical condition filter.
            intervention: Drug/treatment filter.
            sponsor:      Sponsor organisation filter.
            status:       List of status values to filter by.
            page_token:   Cursor for the next page of results.

        Returns:
            Dictionary of query parameters ready to send to the API.
        """

        params: dict[str, Any] = {
            "pageSize": PAGE_SIZE,
            "format": "json",
        }

        if condition:
            params["query.cond"] = condition

        if intervention:
            params["query.intr"] = intervention

        if sponsor:
            params["query.spons"] = sponsor

        if status:
            params["filter.overallStatus"] = "|".join(status)

        if page_token:
            params["pageToken"] = page_token

        return params


    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
        ),
    )
    async def _fetch_page(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """
        Makes one GET request to the /studies endpoint.

        This is decorated with @retry from tenacity, which means
        if it fails with a Timeout or ConnectionError, tenacity
        automatically calls it again : up to MAX_RETRIES times.

        Args:
            params: The query parameters built by _build_search_params.

        Returns:
            The JSON response as a Python dictionary.
            None if all retry attempts failed.
        """

        def _get():
            return self._session.get(
                f"{BASE_URL}/studies",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

        try:
            response = await asyncio.to_thread(_get)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning(
                f"Request timed out after {REQUEST_TIMEOUT}s — retrying..."
            )
            raise

        except requests.exceptions.ConnectionError:
            logger.warning("Connection error — retrying...")
            raise

        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP error from API | "
                f"status={e.response.status_code} | "
                f"url={e.response.url}"
            )
            return None

        except Exception as e:
            logger.error(
                f"Unexpected error fetching page | "
                f"error={e}"
            )
            return None