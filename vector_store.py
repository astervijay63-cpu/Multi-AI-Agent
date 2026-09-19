import asyncio
import asyncpg
import json
from typing import Any
from processing.embedder import EmbeddedChunk
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

# configuration
POOL_MIN_SIZE = 2
POOL_MAX_SIZE = 10
TOP_K_DEFAULT = 5


class VectorStore:
    """
    Saves EmbeddedChunks to Postgres and enables semantic search
    over them using pgvector's cosine similarity operator.

    LIFECYCLE — always follow this order:
        vs = VectorStore()   # create
        await vs.init()      # connect to database
        # ... use it ...
        await vs.close()     # disconnect cleanly

    OR use it as an async context manager:
        async with VectorStore() as vs:
            await vs.save_embedded_chunks(chunks)
            results = await vs.search(query_embedding)
    """

    def __init__(self):
        self._pool: asyncpg.Pool | None = None

    @property
    def is_initialised(self) -> bool:
        """True once init() has created the connection pool."""
        return self._pool is not None

    @property
    def pool(self) -> asyncpg.Pool:
        """
        Returns the active connection pool, raising a clear error
        if init() has not been called yet. Narrows the type from
        asyncpg.Pool | None to asyncpg.Pool for the type checker.
        """
        if self._pool is None:
            raise RuntimeError(
                "VectorStore is not initialised — call await vs.init() "
                "first, or use 'async with VectorStore() as vs'."
            )
        return self._pool

    async def __aenter__(self) -> "VectorStore":
        """Allows using VectorStore with 'async with' pattern."""
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the connection pool when exiting 'async with'."""
        await self.close()

    async def init(self) -> None:
        """
        Creates the asyncpg connection pool and registers the
        pgvector codec so Python can read and write VECTOR columns.

        MUST be called before any other method.
        This is where we actually connect to Postgres.
        """

        logger.info(
            f"Connecting to Postgres | "
            f"host={settings.db_host} | "
            f"database={settings.db_name}"
        )

        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=int(settings.db_port),
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            ssl="require" if settings.db_ssl else None,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            init=self._init_connection,
        )

        logger.info("Connection pool created successfully")

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """
        Runs automatically on every new database connection.

        This is where we register the pgvector codec —
        the translator that teaches asyncpg how to convert
        between Python lists and PostgreSQL VECTOR columns.

        WITHOUT THIS:
            Saving a chunk → TypeError: cannot convert list to VECTOR
            Reading a chunk → asyncpg.exceptions.UndefinedTypeError

        WITH THIS:
            Python [0.023, -0.041, 0.891, ...] ↔ PostgreSQL VECTOR(1536)
            The conversion happens automatically, invisibly.

        Args:
            conn: A fresh asyncpg connection, just created by the pool.
        """

        await conn.set_type_codec(
            "vector",
            encoder=lambda v: json.dumps(v),
            decoder=lambda v: json.loads(v),
            schema="public",
            format="text",
        )

    async def close(self) -> None:
        """
        Gracefully closes all database connections in the pool.
        Always call this when you are done with the VectorStore.
        Leaving connections open wastes database resources.
        """

        if self._pool:
            await self._pool.close()
            logger.info("Connection pool closed")

    async def save_embedded_chunks(
        self,
        chunks: list[EmbeddedChunk],
    ) -> int:
        """
        Saves a list of EmbeddedChunks into the chunks table.

        Uses INSERT ... ON CONFLICT DO NOTHING so it is safe
        to run multiple times — duplicate chunks are silently
        skipped instead of causing an error.

        Args:
            chunks: List of EmbeddedChunk objects to save.

        Returns:
            Number of chunks successfully saved.
        """

        if not chunks:
            logger.warning("save_embedded_chunks called with empty list")
            return 0

        saved_count = 0

        async with self.pool.acquire() as conn:
            for chunk in chunks:
                try:
                    await conn.execute(
                        """
                        INSERT INTO chunks
                            (nct_id, chunk_text, embedding, chunk_index, source)
                        VALUES
                            ($1, $2, $3, $4, $5)
                        ON CONFLICT DO NOTHING
                        """,
                        chunk.nct_id,
                        chunk.chunk_text,
                        chunk.embedding,
                        chunk.chunk_index,
                        chunk.source,
                    )

                    saved_count += 1

                except Exception as e:
                    logger.error(
                        f"Failed to save chunk | "
                        f"chunk_id={chunk.chunk_id} | "
                        f"error={e}"
                    )

        logger.info(
            f"Chunks saved | "
            f"saved={saved_count} | "
            f"total_input={len(chunks)} | "
            f"skipped={len(chunks) - saved_count}"
        )

        return saved_count

    async def save_study(
        self,
        study_data: dict[str, Any],
    ) -> None:
        """
        Saves one study record into the studies table.

        Uses INSERT ... ON CONFLICT (nct_id) DO UPDATE so that
        if a study already exists, its fields get refreshed
        with the latest data instead of being skipped.

        Args:
            study_data: Dictionary of study fields to save.
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO studies
                    (nct_id, title, sponsor, phase, status,
                     conditions, interventions, primary_outcome,
                     secondary_outcomes, start_date, completion_date,
                     results_posted, enrollment, storage_path)
                VALUES
                    ($1, $2, $3, $4, $5,
                     $6, $7, $8,
                     $9, $10, $11,
                     $12, $13, $14)
                ON CONFLICT (nct_id) DO UPDATE SET
                    title            = EXCLUDED.title,
                    sponsor          = EXCLUDED.sponsor,
                    phase            = EXCLUDED.phase,
                    status           = EXCLUDED.status,
                    conditions       = EXCLUDED.conditions,
                    interventions    = EXCLUDED.interventions,
                    primary_outcome  = EXCLUDED.primary_outcome,
                    secondary_outcomes = EXCLUDED.secondary_outcomes,
                    start_date       = EXCLUDED.start_date,
                    completion_date  = EXCLUDED.completion_date,
                    results_posted   = EXCLUDED.results_posted,
                    enrollment       = EXCLUDED.enrollment,
                    storage_path     = EXCLUDED.storage_path
                """,
                study_data.get("nct_id"),
                study_data.get("title"),
                study_data.get("sponsor"),
                study_data.get("phase"),
                study_data.get("status"),
                study_data.get("conditions", []),
                study_data.get("interventions", []),
                study_data.get("primary_outcome"),
                study_data.get("secondary_outcomes", []),
                study_data.get("start_date"),
                study_data.get("completion_date"),
                study_data.get("results_posted"),
                study_data.get("enrollment"),
                study_data.get("storage_path"),
            )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_DEFAULT,
        source_filter: str | None = None,
        nct_id_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Finds the most semantically similar chunks to a query embedding.

        Uses pgvector's cosine distance operator (<=>)  to compare
        the query embedding against every stored chunk embedding
        and returns the TOP_K closest ones.

        This is the method every agent calls when it needs context.
        It is the bridge between a natural language question and
        the relevant chunks stored in Postgres.

        Args:
            query_embedding:  The search query as 1536 numbers.
            top_k:            How many results to return.
            source_filter:    Optional filter by source type.
            nct_id_filter:    Optional filter by specific study.

        Returns:
            List of dictionaries, each containing:
            - nct_id:      Which study this chunk belongs to
            - chunk_text:  The actual text content
            - chunk_index: Position in the original document
            - source:      "study" or "paper"
            - distance:    Cosine distance (lower = more similar)
                           0.0 = identical meaning
                           1.0 = completely different meaning
                           2.0 = opposite meaning
        """

        conditions = []

        params: list[Any] = [query_embedding]

        param_count = 1

        if source_filter:
            param_count += 1
            conditions.append(f"source = ${param_count}")
            params.append(source_filter)

        if nct_id_filter:
            param_count += 1
            conditions.append(f"nct_id = ${param_count}")
            params.append(nct_id_filter)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        param_count += 1
        params.append(top_k)

        query = f"""
            SELECT
                nct_id,
                chunk_text,
                chunk_index,
                source,
                embedding <=> $1 AS distance
            FROM chunks
            {where_clause}
            ORDER BY distance ASC
            LIMIT ${param_count}
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = [dict(row) for row in rows]

        logger.info(
            f"Semantic search complete | "
            f"results_found={len(results)} | "
            f"top_k={top_k} | "
            f"source_filter={source_filter} | "
            f"nct_id_filter={nct_id_filter}"
        )

        return results

    async def get_chunk_count(self) -> int:
        """
        Returns the total number of chunks currently in the database.
        Used by run_processing.py to report progress after saving.

        Returns:
            Total count of rows in the chunks table.
        """

        async with self.pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM chunks")

        logger.info(f"Total chunks in database: {result}")
        return result

    async def study_exists(self, nct_id: str) -> bool:
        """
        Checks if a study already has chunks saved in the database.

        Used by run_processing.py to skip studies that were already
        processed in a previous run — avoids duplicate work.

        Args:
            nct_id: The study to check.

        Returns:
            True if this study already has chunks in the database.
            False if it has not been processed yet.
        """

        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE nct_id = $1",
                nct_id,
            )

        exists = count > 0

        return exists

    async def get_chunks_for_study(
        self,
        nct_id: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieves every chunk belonging to one specific study.

        Unlike search(), this does NOT rank by similarity — there is
        no query to compare against. It simply returns all chunks for
        the given nct_id, ordered the way they appeared in the source
        document, so an agent can read the study's full content in order.

        Args:
            nct_id: The study to retrieve chunks for.

        Returns:
            List of dictionaries, each containing:
            - nct_id:      Always equal to the requested nct_id
            - chunk_text:  The actual text content
            - chunk_index: Position in the original document
            - source:      "study" or "paper"
            Empty list if the study has no chunks saved yet.
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    nct_id,
                    chunk_text,
                    chunk_index,
                    source
                FROM chunks
                WHERE nct_id = $1
                ORDER BY chunk_index ASC
                """,
                nct_id,
            )

        results = [dict(row) for row in rows]

        logger.info(
            f"get_chunks_for_study complete | "
            f"nct_id={nct_id} | "
            f"chunks_found={len(results)}"
        )

        return results