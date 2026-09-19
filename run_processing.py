import asyncio
from ingestion.document_parser import ParsedStudy
from ingestion.backblaze_store import B2Store
from processing.chunker import Chunker
from processing.embedder import Embedder
from processing.vector_store import VectorStore

from config.logging_config import setup_logging
logger = setup_logging(__name__)


async def run_processing():
    """
    Runs the complete processing pipeline from start to finish.

    Reads cleaned studies from B2, chunks them, embeds them via
    OpenAI, and stores everything in Postgres for agent search.
    """
    b2_store = B2Store()
    chunker = Chunker()
    embedder = Embedder()
    logger.info("Loading parsed studies from B2...")
    nct_ids = await b2_store.list_processed_studies()
    logger.info(f"Found {len(nct_ids)} studies in B2")

    studies : list[ParsedStudy] = []
    for nct_id in nct_ids:
        study = await b2_store.load_parsed_study(nct_id)
        if study:
            studies.append(study)
        logger.info(f"Loaded {len(studies)} studies successfully")

    async with VectorStore() as vector_store:
        logger.info("Saving study metadata to Postgres...")

        studies_saved = 0
        for study in studies:
            success = await vector_store.save_study(
                study_data={
                    "nct_id":             study.nct_id,
                    "title":              study.title,
                    "sponsor":            study.sponsor,
                    "phase":              study.phase,
                    "status":             study.status,
                    "conditions":         study.conditions,
                    "interventions":      study.interventions,
                    "primary_outcome":    study.primary_outcome,
                    "secondary_outcomes": study.secondary_outcomes,
                    "start_date":         study.start_date,
                    "completion_date":    study.completion_date,
                    "results_posted":     study.results_posted,
                    "enrollment":         study.enrollment,
                    "storage_path":       f"processed/studies/{study.nct_id}.json",
                }
            )
            if success:
                studies_saved += 1

        logger.info(
            f"Studies saved to Postgres | "
            f"saved={studies_saved} | "
            f"total={len(studies)}"
        )

        logger.info("Chunking studies...")

        all_chunks = chunker.chunk_studies(studies)

        logger.info(f"Total chunks created: {len(all_chunks)}")

        logger.info("Embedding chunks via OpenAI...")

        embedded_chunks = await embedder.embed_chunks(all_chunks)

        logger.info(f"Total chunks embedded: {len(embedded_chunks)}")

        logger.info("Saving embedded chunks to Postgres...")

        chunks_stored = await vector_store.save_embedded_chunks(embedded_chunks)

        logger.info("=" * 60)
        logger.info("Processing complete")
        logger.info(f"Studies processed : {len(studies)}")
        logger.info(f"Chunks created    : {len(all_chunks)}")
        logger.info(f"Chunks embedded   : {len(embedded_chunks)}")
        logger.info(f"Chunks stored     : {chunks_stored}")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_processing())