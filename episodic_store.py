import json
import uuid
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncpg
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

class EpisodicStore:
    """
    Stores and retrieves agent reasoning sessions as episodes.

    Each episode is one agent's reasoning session — what it
    investigated, what it found, and what it concluded.
    Episodes are embedded and stored so future sessions can
    search through past findings by meaning.

    Usage:
        store = EpisodicStore()

        # Save what an agent found
        await store.save_episode(
            agent_name="missing_results_agent",
            nct_id="NCT04788680",
            content="Novo Nordisk trial completed 2019. Results never posted.",
            outcome="signal_generated"
        )

        # Search past episodes by meaning
        past = await store.search_episodes(
            query="sponsor never posted results",
            agent_name="missing_results_agent",
            top_k=3
        )
    """

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        self._embedding_model = settings.google_embedding_model
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self._embedding_model,
            api_key=settings.google_api_key,
            # This must match the column and processing embedder dimensions.
            output_dimensionality=1536,
        )

        logger.info("EpisodicStore initialised")

    @property
    def pool(self) -> asyncpg.Pool:
        """
        Returns the active connection pool, raising a clear error
        if the pool has not been created yet. Narrows the type from
        asyncpg.Pool | None to asyncpg.Pool for the type checker.
        """
        if self._pool is None:
            raise RuntimeError(
                "EpisodicStore pool is not initialised — call "
                "await store._ensure_pool() first, or call a public "
                "method (save_episode, search_episodes, etc.) which "
                "does this automatically."
            )
        return self._pool

    async def _ensure_pool(self) -> None:
        """
        Makes sure the database connection pool is open.

        WHAT IS LAZY INITIALISATION?
        Instead of opening the database connection the moment
        EpisodicStore() is created, we wait until someone actually
        tries to use it. This is called "lazy initialisation."

        WHY DO WE DO THIS?
        When MOSAIC starts up, it imports many classes including
        EpisodicStore. If we opened the database connection in
        __init__, every import would immediately try to connect
        to Postgres — even if that class is never actually used
        in that run. Lazy initialisation avoids wasted connections.

        This method is called at the START of every public method
        (save_episode, search_episodes, etc.) to guarantee the
        pool is open before we try to use it.
        """

        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            ssl="require" if settings.db_ssl else None,
            min_size=1,
            max_size=5,
            init=self._init_connection,
        )

        logger.info("EpisodicStore pool created")

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """
        Registers the custom pgvector VECTOR codec on a connection.

        WHAT IS A CODEC?
        A codec is a pair of functions:
          ENCODER: converts Python data → database format
          DECODER: converts database format → Python data

        WHY DO WE NEED A CUSTOM CODEC FOR VECTOR?
        PostgreSQL knows about the VECTOR type (added by pgvector).
        But asyncpg is a generic driver — it only knows about
        standard PostgreSQL types like TEXT, INTEGER, FLOAT etc.
        It has NO idea what to do with VECTOR.

        Without registering this codec:
          - Writing: asyncpg cannot convert our Python list of
            floats into the format pgvector expects → error
          - Reading: asyncpg cannot convert the VECTOR value it
            gets back from the database into a Python list → error

        With this codec registered:
          - Writing: [0.023, -0.041, 0.891] → "[0.023,-0.041,0.891]"
          - Reading: "[0.023,-0.041,0.891]" → [0.023, -0.041, 0.891]

        asyncpg calls this function automatically on every new
        connection it creates, via the init= parameter above.

        Args:
            conn: One fresh database connection from the pool.
                  asyncpg passes this automatically.
        """

        await conn.set_type_codec(
            "vector",
            encoder=lambda v: json.dumps(v),
            decoder=lambda v: json.loads(v),
            schema="public",
        )

    async def _embed(self, text: str) -> list[float]:
        """
        Converts any text string into a 1536-number vector embedding.

        WHAT IS AN EMBEDDING — ONE MORE TIME SIMPLY:
        An embedding is a position on a giant meaning map.
        Similar texts end up close together on that map.
        "Results never posted" and "outcomes not published" would
        be very close together even though the words are different.

        We use this method in TWO places:
        1. When SAVING an episode — embed the content so it can
           be found by semantic search later.
        2. When SEARCHING episodes — embed the query so we can
           compare it against stored episode embeddings.

        Using the SAME model for both is critical — embeddings
        from different models cannot be compared meaningfully.

        Args:
            text: Any text string to convert to an embedding.

        Returns:
            A list of exactly 1536 floating point numbers.
            Example (first 3 of 1536): [0.023, -0.041, 0.891, ...]
        """

        return await self._embeddings.aembed_query(text)

    async def save_episode(
        self,
        agent_name: str,
        content: str,
        nct_id: str | None = None,
        outcome: str | None = None,
    ) -> str:
        """
        Saves one agent reasoning session as an episode in Postgres.

        WHAT HAPPENS INSIDE THIS METHOD:
        1. Makes sure the database connection is open
        2. Generates a unique ID for this episode
        3. Converts the content to a 1536-number embedding
        4. Inserts all of this into the episodes table
        5. Returns the episode_id

        Args:
            agent_name: Which agent is saving this.
            content:    What the agent found — plain text.
            nct_id:     Which study this is about (optional).
            outcome:    What happened as a result (optional).

        Returns:
            episode_id — the unique ID of the saved episode.
        """

        await self._ensure_pool()

        episode_id = str(uuid.uuid4())

        embedding = await self._embed(content)

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO episodes (
                    episode_id,
                    agent_name,
                    nct_id,
                    content,
                    outcome,
                    embedding,
                    created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                episode_id,
                agent_name,
                nct_id,
                content,
                outcome,
                embedding,
                datetime.utcnow(),
            )

        logger.info(
            f"Episode saved | "
            f"agent={agent_name} | "
            f"nct_id={nct_id} | "
            f"outcome={outcome} | "
            f"episode_id={episode_id}"
        )

        return episode_id

    async def search_episodes(
        self,
        query: str,
        agent_name: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[dict]:
        """
        Searches past episodes by semantic similarity to a query.

        WHAT HAPPENS INSIDE THIS METHOD:
        1. Makes sure the database connection is open
        2. Converts the query text to a 1536-number embedding
        3. Builds a SQL query with optional filters
        4. Uses pgvector's <=> operator to find similar episodes
        5. Returns the top_k most similar episodes as dictionaries

        Think of this as the agent "consulting its memory" before
        starting a new investigation.

        Args:
            query:          What to search for in past episodes.
            agent_name:     Optional filter — only this agent's episodes.
            top_k:          How many results to return.
            min_similarity: Minimum similarity score (0.0 to 1.0).

        Returns:
            List of episode dictionaries, most similar first.
            Each dict contains: episode_id, agent_name, nct_id,
                               content, outcome, similarity, created_at
            Empty list if no relevant past episodes found.
        """

        await self._ensure_pool()

        query_embedding = await self._embed(query)

        sql = """
            SELECT
                episode_id,
                agent_name,
                nct_id,
                content,
                outcome,
                created_at,
                1 - (embedding <=> $1) AS similarity
            FROM episodes
            WHERE 1 - (embedding <=> $1) >= $2
        """

        params: list = [query_embedding, min_similarity]

        param_idx = 3

        if agent_name:
            sql += f" AND agent_name = ${param_idx}"

            params.append(agent_name)

            param_idx += 1

        sql += f"""
            ORDER BY embedding <=> $1
            LIMIT ${param_idx}
        """
        params.append(top_k)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        episodes = [
            {
                "episode_id": row["episode_id"],
                "agent_name": row["agent_name"],
                "nct_id":     row["nct_id"],
                "content":    row["content"],
                "outcome":    row["outcome"],
                "similarity": round(float(row["similarity"]), 3),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

        logger.info(
            f"Episode search complete | "
            f"query='{query[:50]}...' | "
            f"agent_filter={agent_name} | "
            f"results_found={len(episodes)}"
        )

        return episodes

    async def get_recent_episodes(
        self,
        agent_name: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Returns the most recent episodes, newest first.

        Unlike search_episodes() which finds episodes by MEANING,
        this method finds episodes by TIME — just the most recent ones.

        Used by the API endpoint GET /api/v1/memory/episodes so
        analysts can browse what the agents have been doing recently.

        Args:
            agent_name: Optional — filter to one agent's episodes.
            limit:      Maximum number of episodes to return.

        Returns:
            List of episode dicts ordered newest first.
        """

        await self._ensure_pool()

        if agent_name:
            sql = """
                SELECT episode_id, agent_name, nct_id,
                       content, outcome, created_at
                FROM episodes
                WHERE agent_name = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            params = [agent_name, limit]
        else:
            sql = """
                SELECT episode_id, agent_name, nct_id,
                       content, outcome, created_at
                FROM episodes
                ORDER BY created_at DESC
                LIMIT $1
            """
            params = [limit]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            {
                "episode_id": row["episode_id"],
                "agent_name": row["agent_name"],
                "nct_id":     row["nct_id"],
                "content":    row["content"],
                "outcome":    row["outcome"],
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    async def count_episodes(
        self,
        agent_name: str | None = None,
    ) -> int:
        """
        Returns the total number of episodes stored.

        Used by the health check endpoint to show how much memory
        the system has accumulated — how many past sessions exist.

        Args:
            agent_name: Optional — count only this agent's episodes.

        Returns:
            Integer count. Example: 47 (meaning 47 episodes stored.)
        """

        await self._ensure_pool()

        async with self.pool.acquire() as conn:

            if agent_name:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM episodes WHERE agent_name = $1",
                    agent_name,
                )
            else:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM episodes"
                )

        return count or 0

    async def close(self) -> None:
        """
        Closes the connection pool and releases all connections.

        Call this when the application shuts down cleanly.
        Without closing, connections may stay open on Postgres
        unnecessarily — wasting resources and potentially hitting
        connection limits.

        In FastAPI, this is called in the lifespan shutdown handler.
        """

        if self._pool:
            await self._pool.close()

            self._pool = None

            logger.info("EpisodicStore pool closed")
