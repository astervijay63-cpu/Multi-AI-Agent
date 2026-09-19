from dataclasses import dataclass
from ingestion.document_parser import ParsedStudy, ParsedPaper
from config.logging_config import setup_logging

logger = setup_logging(__name__)

# configuration
CHUNK_SIZE = 500
OVERLAP_SIZE = 50


@dataclass
class TextChunk:
    """
    One chunk of text from a study or paper, ready to be embedded.

    Fields:
        chunk_id:    Unique identifier for this specific chunk.
                     Format: NCT_ID_chunk_0, NCT_ID_chunk_1, etc.
        nct_id:      Which study this chunk belongs to.
        chunk_text:  The actual text content of this chunk.
        chunk_index: Position of this chunk in the document.
                     0 = first chunk, 1 = second chunk, etc.
        source:      Where this chunk came from.
                     "study"  = from a ClinicalTrials.gov record
                     "paper"  = from a PubMed research paper
        word_count:  How many words are in this chunk.
    """

    chunk_id:    str
    nct_id:      str
    chunk_text:  str
    chunk_index: int
    source:      str
    word_count:  int


class Chunker:
    """
    Splits study and paper documents into overlapping text chunks.

    Usage:
        chunker = Chunker()
        chunks = chunker.chunk_study(parsed_study)
        chunks = chunker.chunk_paper(parsed_paper)
    """

    def chunk_study(self, study: ParsedStudy) -> list[TextChunk]:
        """
        Takes one ParsedStudy and produces a list of TextChunks.

        First we BUILD the full text by combining all the study's
        important fields into one long string, with clear labels
        so the embedding model knows what each section means.

        Then we SPLIT that long string into overlapping chunks.

        Args:
            study: A clean ParsedStudy object from document_parser.py

        Returns:
            A list of TextChunk objects, ready for embedding.
        """

        full_text = self._build_study_text(study)

        chunks = self._split_into_chunks(
            text=full_text,
            nct_id=study.nct_id,
            source="study",
        )

        logger.info(
            f"Chunked study | "
            f"nct_id={study.nct_id} | "
            f"chunks_produced={len(chunks)}"
        )

        return chunks

    def chunk_paper(self, paper: ParsedPaper) -> list[TextChunk]:
        """
        Takes one ParsedPaper and produces a list of TextChunks.

        Same two-step process as chunk_study:
        1. Build the full text from all the paper's fields.
        2. Split into overlapping chunks.

        Args:
            paper: A clean ParsedPaper object from document_parser.py

        Returns:
            A list of TextChunk objects, ready for embedding.
        """

        full_text = self._build_paper_text(paper)
        chunks    = self._split_into_chunks(
            text=full_text,
            nct_id=paper.pmid,
            source="paper",
        )

        logger.info(
            f"Chunked paper | "
            f"pmid={paper.pmid} | "
            f"chunks_produced={len(chunks)}"
        )

        return chunks

    def chunk_studies(self, studies: list[ParsedStudy]) -> list[TextChunk]:
        """
        Chunks a whole list of studies in one call.
        Returns ALL chunks from ALL studies as one flat list.

        Args:
            studies: List of ParsedStudy objects to chunk.

        Returns:
            Flat list of all TextChunks from all studies combined.
        """

        all_chunks: list[TextChunk] = []

        for study in studies:
            chunks = self.chunk_study(study)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked all studies | "
            f"studies={len(studies)} | "
            f"total_chunks={len(all_chunks)}"
        )

        return all_chunks

    def chunk_papers(self, papers: list[ParsedPaper]) -> list[TextChunk]:
        """
        Chunks a whole list of papers in one call.
        Same pattern as chunk_studies — returns one flat list.

        Args:
            papers: List of ParsedPaper objects to chunk.

        Returns:
            Flat list of all TextChunks from all papers combined.
        """

        all_chunks: list[TextChunk] = []

        for paper in papers:
            chunks = self.chunk_paper(paper)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked all papers | "
            f"papers={len(papers)} | "
            f"total_chunks={len(all_chunks)}"
        )

        return all_chunks

    def _build_study_text(self, study: ParsedStudy) -> str:
        """
        Combines all a study's fields into one labelled text block.

        Labelling each field gives the embedding model context —
        "SPONSOR: Novo Nordisk" is more informative than the bare
        name. Plain labelled text also embeds better than raw JSON,
        which carries structural noise the model does not need.

        Args:
            study: The ParsedStudy to convert to text.

        Returns:
            One long string containing all the study's key fields,
            each clearly labelled on its own line.
        """

        sections = []

        sections.append(f"NCT ID: {study.nct_id}")
        sections.append(f"TITLE: {study.title}")
        sections.append(f"SPONSOR: {study.sponsor}")
        sections.append(f"PHASE: {study.phase}")
        sections.append(f"STATUS: {study.status}")

        if study.conditions:
            sections.append(
                f"CONDITIONS: {', '.join(study.conditions)}"
            )

        if study.interventions:
            sections.append(
                f"INTERVENTIONS: {', '.join(study.interventions)}"
            )

        if study.primary_outcome:
            sections.append(
                f"PRIMARY OUTCOME: {study.primary_outcome}"
            )

        if study.secondary_outcomes:
            sections.append(
                f"SECONDARY OUTCOMES: {'; '.join(study.secondary_outcomes)}"
            )

        if study.start_date:
            sections.append(f"START DATE: {study.start_date}")

        if study.completion_date:
            sections.append(f"COMPLETION DATE: {study.completion_date}")

        sections.append(
            f"RESULTS POSTED: {'YES' if study.results_posted else 'NO'}"
        )

        if study.enrollment:
            sections.append(f"ENROLLMENT: {study.enrollment} participants")

        if study.protocol_amendments:
            sections.append(
                f"PROTOCOL AMENDMENTS: "
                f"{len(study.protocol_amendments)} amendment(s) filed"
            )

        return "\n".join(sections)

    def _build_paper_text(self, paper: ParsedPaper) -> str:
        """
        Combines all a paper's fields into one labelled text block.
        Same idea as _build_study_text but for PubMed papers.

        Only the first five authors are included, and the abstract
        is placed last because it is by far the longest field.

        Args:
            paper: The ParsedPaper to convert to text.

        Returns:
            One long string with all the paper's key fields labelled.
        """

        sections = []

        sections.append(f"PMID: {paper.pmid}")
        sections.append(f"TITLE: {paper.title}")

        if paper.journal:
            sections.append(f"JOURNAL: {paper.journal}")

        if paper.pub_date:
            sections.append(f"PUBLICATION DATE: {paper.pub_date}")

        if paper.authors:
            sections.append(
                f"AUTHORS: {', '.join(paper.authors[:5])}"
            )

        if paper.nct_ids_referenced:
            sections.append(
                f"CLINICAL TRIALS REFERENCED: "
                f"{', '.join(paper.nct_ids_referenced)}"
            )

        if paper.abstract:
            sections.append(f"ABSTRACT: {paper.abstract}")

        return "\n".join(sections)

    def _split_into_chunks(
        self,
        text: str,
        nct_id: str,
        source: str,
    ) -> list[TextChunk]:
        """
        The core splitting algorithm. Splits one long text into
        overlapping chunks of CHUNK_SIZE words each.

        HOW IT WORKS:
        1. Split the full text into individual words
        2. Use a sliding window of CHUNK_SIZE words
        3. Slide forward by (CHUNK_SIZE - OVERLAP_SIZE) words each time
        4. This creates the overlap — each chunk shares OVERLAP_SIZE
           words with the next one

        Args:
            text:   The full text to split.
            nct_id: Source document ID for naming chunks.
            source: "study" or "paper" for tagging chunks.

        Returns:
            List of TextChunk objects.
        """

        words = text.split()

        if not words:
            logger.warning(
                f"Empty text — no chunks produced | "
                f"nct_id={nct_id} | source={source}"
            )
            return []

        chunks:     list[TextChunk] = []
        chunk_index = 0

        step = CHUNK_SIZE - OVERLAP_SIZE

        for start in range(0, len(words), step):
            end = start + CHUNK_SIZE
            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunk_text = " ".join(chunk_words)

            chunk = TextChunk(
                chunk_id=f"{nct_id}_chunk_{chunk_index}",
                nct_id=nct_id,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                source=source,
                word_count=len(chunk_words),
            )

            chunks.append(chunk)
            chunk_index += 1

        logger.info(
            f"Split complete | "
            f"nct_id={nct_id} | "
            f"total_words={len(words)} | "
            f"chunks={len(chunks)} | "
            f"chunk_size={CHUNK_SIZE} | "
            f"overlap={OVERLAP_SIZE}"
        )

        return chunks