# Transforms raw API responses into structured, clean internal responses 
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from config.logging_config import setup_logging

logger = setup_logging(__name__)


class ParsedStudy(BaseModel):
    """
    A clinical trial study, cleaned and structured.

    This is what a "study" means everywhere else in our codebase.
    The chunker reads this. The vector store reads this.
    The agents reason about this. Nobody touches raw API data
    except this one file.
    """

    nct_id: str
    title: str
    sponsor: str
    phase: str
    status: str
    conditions: list[str]
    interventions: list[str]
    primary_outcome: str
    secondary_outcomes: list[str]
    start_date: str
    completion_date: str
    results_posted: bool
    enrollment: int
    protocol_amendments: list[dict[str, Any]]
    raw_data: dict[str, Any]
    parsed_at: str


class ParsedPaper(BaseModel):
    """
    A PubMed research paper, cleaned and structured.

    Same idea as ParsedStudy — this is what a "paper" means
    everywhere else in our codebase.
    """

    pmid: str
    title: str
    abstract: str
    journal: str
    pub_date: str
    authors: list[str]
    nct_ids_referenced: list[str]
    source: str = "pubmed"
    word_count: int
    parsed_at: str


class DocumentParser:
    """
    Converts raw API data into clean ParsedStudy and ParsedPaper objects.

    Usage:
        parser = DocumentParser()
        study = parser.parse_study(raw_study_dict)
        paper = parser.parse_paper(raw_paper_dict)
    """

    def parse_study(self, raw: dict[str, Any]) -> ParsedStudy | None:
        """
        Cleans one raw ClinicalTrials.gov study record.

        The ClinicalTrials.gov API nests everything very deeply —
        a field like "title" might be 4 or 5 levels deep inside
        the raw dictionary. This method digs through that nesting
        and pulls out only what we need.

        Args:
            raw: One raw study dictionary, exactly as the API returned it.

        Returns:
            A ParsedStudy if everything worked.
            None if something essential was missing or broken.
        """

        try:
            protocol = raw.get("protocolSection", {})

            id_module          = protocol.get("identificationModule", {})
            status_module      = protocol.get("statusModule", {})
            sponsor_module     = protocol.get("sponsorCollaboratorsModule", {})
            conditions_module  = protocol.get("conditionsModule", {})
            design_module      = protocol.get("designModule", {})
            outcomes_module    = protocol.get("outcomesModule", {})
            interventions_mod  = protocol.get("armsInterventionsModule", {})

            results_section = raw.get("resultsSection", {})
            has_results     = bool(results_section)

            nct_id = id_module.get("nctId", "")
            if not nct_id:
                logger.warning("Study is missing its NCT ID — skipping")
                return None

            title = (
                id_module.get("officialTitle")
                or id_module.get("briefTitle")
                or ""
            )

            sponsor = (
                sponsor_module
                .get("leadSponsor", {})
                .get("name", "Unknown Sponsor")
            )

            phase = design_module.get("phases", ["NA"])
            phase = phase[0] if phase else "NA"

            status = status_module.get("overallStatus", "UNKNOWN")

            conditions = conditions_module.get("conditions", [])

            interventions = [
                i.get("name", "")
                for i in interventions_mod.get("interventions", [])
                if i.get("name")
            ]

            primary_outcomes_list = outcomes_module.get("primaryOutcomes", [])
            primary_outcome = (
                primary_outcomes_list[0].get("measure", "")
                if primary_outcomes_list
                else ""
            )

            secondary_outcomes = [
                o.get("measure", "")
                for o in outcomes_module.get("secondaryOutcomes", [])
                if o.get("measure")
            ]

            start_date = (
                status_module
                .get("startDateStruct", {})
                .get("date", "")
            )

            completion_date = (
                status_module
                .get("primaryCompletionDateStruct", {})
                .get("date", "")
                or status_module
                .get("completionDateStruct", {})
                .get("date", "")
            )

            enrollment_info = design_module.get("enrollmentInfo", {})
            enrollment = enrollment_info.get("count", 0)
            try:
                enrollment = int(enrollment)
            except (ValueError, TypeError):
                enrollment = 0

            annotations      = raw.get("annotationSection", {})
            amendment_module = annotations.get("annotationModule", {})
            amendments       = amendment_module.get("unpostedAnnotation", {})

            protocol_amendments = []
            if amendments:
                protocol_amendments = [
                    {
                        "date":        amendments.get("unpostedResponsibleParty", ""),
                        "description": str(amendments),
                    }
                ]

            return ParsedStudy(
                nct_id=nct_id,
                title=title,
                sponsor=sponsor,
                phase=phase,
                status=status,
                conditions=conditions,
                interventions=interventions,
                primary_outcome=primary_outcome,
                secondary_outcomes=secondary_outcomes,
                start_date=start_date,
                completion_date=completion_date,
                results_posted=has_results,
                enrollment=enrollment,
                protocol_amendments=protocol_amendments,
                raw_data=raw,
                parsed_at=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            nct_id = raw.get("protocolSection", {}).get(
                "identificationModule", {}
            ).get("nctId", "UNKNOWN")
            logger.error(
                f"Failed to parse study | nct_id={nct_id} | error={e}"
            )
            return None

    def parse_studies(
        self,
        raw_studies: list[dict[str, Any]],
    ) -> list[ParsedStudy]:
        """
        Parses a whole list of raw studies in one call.
        Any study that fails to parse is skipped — not fatal.

        Args:
            raw_studies: List of raw study dicts from the API.

        Returns:
            List of successfully parsed ParsedStudy objects.
            Failed studies are simply not included in the result.
        """

        parsed = []
        failed = 0

        for raw in raw_studies:
            study = self.parse_study(raw)
            if study:
                parsed.append(study)
            else:
                failed += 1

        logger.info(
            f"Parsed studies | "
            f"success={len(parsed)} | "
            f"failed={failed} | "
            f"total={len(raw_studies)}"
        )

        return parsed

    def parse_paper(self, raw: dict[str, Any]) -> ParsedPaper | None:
        """
        Cleans one raw PubMed paper record.

        PubMed papers are simpler than studies — the pubmed_client.py
        file already flattened the XML into a reasonably clean dict.
        This method does the final cleanup and builds the typed object.

        Args:
            raw: One raw paper dictionary from pubmed_client.py.

        Returns:
            A ParsedPaper if everything worked.
            None if something went wrong.
        """

        try:
            abstract = raw.get("abstract", "")
            word_count = len(abstract.split()) if abstract else 0

            return ParsedPaper(
                pmid=raw.get("pmid", ""),
                title=raw.get("title", ""),
                abstract=abstract,
                journal=raw.get("journal", ""),
                pub_date=raw.get("pub_date", ""),
                authors=raw.get("authors", []),
                nct_ids_referenced=raw.get("nct_ids_referenced", []),
                source="pubmed",
                word_count=word_count,
                parsed_at=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            logger.error(
                f"Failed to parse paper | "
                f"pmid={raw.get('pmid', 'UNKNOWN')} | "
                f"error={e}"
            )
            return None

    def parse_papers(
        self,
        raw_papers: list[dict[str, Any]],
    ) -> list[ParsedPaper]:
        """
        Parses a whole list of raw papers in one call.
        Same pattern as parse_studies — failures are skipped, not fatal.

        Args:
            raw_papers: List of raw paper dicts from pubmed_client.py.

        Returns:
            List of successfully parsed ParsedPaper objects.
        """

        parsed = []
        failed = 0

        for raw in raw_papers:
            paper = self.parse_paper(raw)
            if paper:
                parsed.append(paper)
            else:
                failed += 1

        logger.info(
            f"Parsed papers | "
            f"success={len(parsed)} | "
            f"failed={failed} | "
            f"total={len(raw_papers)}"
        )

        return parsed

