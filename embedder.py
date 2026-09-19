import asyncio
from dataclasses import dataclass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from processing.chunker import TextChunk
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

# configuration
BATCH_SIZE = 50
RETRY_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 2


@dataclass
class EmbeddedChunk:
    """
    A TextChunk that has been enriched with its vector embedding.

    This is what gets saved to the Postgres chunks table.
    Every field from TextChunk is carried over, plus one new field:
    embedding — the list of 1536 numbers representing this chunk's meaning.

    Fields:
        chunk_id:    Unique identifier. Example: "NCT04788680_chunk_0"
        nct_id:      Which study or paper this chunk belongs to.
        chunk_text:  The actual text content.
        chunk_index: Position in the original document (0, 1, 2...)
        source:      "study" or "paper"
        word_count:  Number of words in this chunk.
        embedding:   1536 floating point numbers from OpenAI.
                     This is the mathematical representation of meaning.
                     Two chunks that mean similar things will have
                     embeddings that are numerically close to each other.
    """

    chunk_id:    str
    nct_id:      str
    chunk_text:  str
    chunk_index: int
    source:      str
    word_count:  int
    embedding:   list[float]


class Embedder:
    """
    Converts TextChunks into EmbeddedChunks using OpenAI's
    text-embedding-3-small model.

    Processes chunks in batches of BATCH_SIZE for efficiency.
    Automatically retries failed API calls up to RETRY_ATTEMPTS times.

    Usage:
        embedder = Embedder()
        embedded = await embedder.embed_chunks(list_of_text_chunks)
    """

    def __init__(self):
        self._model = settings.google_embedding_model
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self._model,
            api_key=settings.google_api_key,
            # Keep vectors compatible with the existing pgvector schema.
            output_dimensionality=1536,
        )

        logger.info(
            f"Embedder initialised | model={self._model}"
        )

    async def embed_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[EmbeddedChunk]:
        """
        Converts a list of TextChunks into EmbeddedChunks.

        Splits the input into batches of BATCH_SIZE and processes
        each batch with one OpenAI API call. Much more efficient
        than one API call per chunk.

        Args:
            chunks: List of TextChunk objects from chunker.py

        Returns:
            List of EmbeddedChunk objects — same chunks but now
            each has a 1536-number embedding attached.
            Any chunk that fails to embed is skipped — not fatal.
        """

        if not chunks:
            logger.warning("embed_chunks called with empty list — nothing to do")
            return []

        logger.info(
            f"Starting embedding | "
            f"total_chunks={len(chunks)} | "
            f"batch_size={BATCH_SIZE} | "
            f"model={self._model}"
        )

        all_embedded: list[EmbeddedChunk] = []

        batches = self._create_batches(chunks)

        for batch_num, batch in enumerate(batches):
            logger.info(
                f"Embedding batch {batch_num + 1}/{len(batches)} | "
                f"chunks_in_batch={len(batch)}"
            )

            embedded_batch = await self._embed_batch_with_retry(
                batch=batch,
                batch_num=batch_num,
            )

            all_embedded.extend(embedded_batch)

            if batch_num < len(batches) - 1:
                await asyncio.sleep(0.5)

        logger.info(
            f"Embedding complete | "
            f"total_embedded={len(all_embedded)} | "
            f"total_input={len(chunks)} | "
            f"skipped={len(chunks) - len(all_embedded)}"
        )

        return all_embedded

    async def embed_text(self, text: str) -> list[float]:
        """
        Converts one raw string into a single 1536-number embedding.

        Used for search queries — unlike embed_chunks(), there is
        no TextChunk to wrap and no batching to do, just one string
        in and one embedding out.

        Args:
            text: The raw query text to embed.

        Returns:
            A list of 1536 floats — pass this straight into
            VectorStore.search(query_embedding=...).
        """

        return await self._embeddings.aembed_query(text)

    def _create_batches(
        self,
        chunks: list[TextChunk],
    ) -> list[list[TextChunk]]:
        """
        Splits a flat list of chunks into smaller batches.

        Example:
            150 chunks with BATCH_SIZE=50 →
            [[chunk_0..chunk_49], [chunk_50..chunk_99], [chunk_100..chunk_149]]

        Args:
            chunks: The full list of chunks to split.

        Returns:
            A list of lists — each inner list is one batch.
        """

        return [
            chunks[i : i + BATCH_SIZE]
            for i in range(0, len(chunks), BATCH_SIZE)
        ]

    async def _embed_batch_with_retry(
        self,
        batch: list[TextChunk],
        batch_num: int,
    ) -> list[EmbeddedChunk]:
        """
        Embeds one batch of chunks, retrying on failure.

        Attempts the embedding up to RETRY_ATTEMPTS times.
        Waits RETRY_SLEEP_SECONDS between each attempt.
        Returns an empty list if all attempts fail — the pipeline
        continues with the remaining batches rather than crashing.

        Args:
            batch:     One batch of TextChunks to embed.
            batch_num: The batch number (for logging only).

        Returns:
            List of EmbeddedChunks for this batch.
            Empty list if all retry attempts failed.
        """

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                return await self._embed_batch(batch=batch)

            except Exception as e:
                if attempt < RETRY_ATTEMPTS:
                    logger.warning(
                        f"Embedding failed | "
                        f"batch={batch_num + 1} | "
                        f"attempt={attempt}/{RETRY_ATTEMPTS} | "
                        f"error={e} | "
                        f"retrying in {RETRY_SLEEP_SECONDS}s..."
                    )
                    await asyncio.sleep(RETRY_SLEEP_SECONDS)

                else:
                    logger.error(
                        f"Embedding failed after {RETRY_ATTEMPTS} attempts | "
                        f"batch={batch_num + 1} | "
                        f"error={e} | "
                        f"skipping this batch"
                    )
                    return []

        return []

    async def _embed_batch(
        self,
        batch: list[TextChunk],
    ) -> list[EmbeddedChunk]:
        """
        Makes ONE OpenAI API call to embed an entire batch of chunks.

        This is the method that actually talks to OpenAI.
        It sends up to BATCH_SIZE chunk texts in one request
        and gets back one embedding per chunk.

        Args:
            batch: One batch of TextChunks (up to BATCH_SIZE).

        Returns:
            List of EmbeddedChunks with embeddings attached.

        Raises:
            Exception: If the OpenAI API call fails.
                       The caller (_embed_batch_with_retry) handles this.
        """

        texts = [chunk.chunk_text for chunk in batch]

        vectors = await self._embeddings.aembed_documents(texts)

        embedded_chunks: list[EmbeddedChunk] = []

        for i, chunk in enumerate(batch):
            embedding_vector = vectors[i]

            embedded_chunk = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                nct_id=chunk.nct_id,
                chunk_text=chunk.chunk_text,
                chunk_index=chunk.chunk_index,
                source=chunk.source,
                word_count=chunk.word_count,
                embedding=embedding_vector,
            )

            embedded_chunks.append(embedded_chunk)

        logger.info(
            f"Batch embedded successfully | "
            f"chunks={len(embedded_chunks)} | "
            f"embedding_dims={len(embedded_chunks[0].embedding) if embedded_chunks else 0}"
        )

        return embedded_chunks
