"""
Sponsor knowledge base for MOSAIC agents.

Semantic memory holds accumulated FACTS about the world — in this
case, a credibility profile for every research sponsor MOSAIC has
ever encountered. Every time MOSAIC analyses a study, it updates the
relevant sponsor's profile: posting results on time raises
credibility, missing results lowers it, outcome switching increases
the broken-promises count, and silent delays raise the average delay.
Over time this builds a data-driven picture of each sponsor's
behaviour, rather than one based on reputation.
"""

import asyncpg
from datetime import datetime
from typing import Any
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


class SemanticStore:
    """
    Manages the sponsor knowledge base — credibility profiles built
    up over time as MOSAIC analyses more studies.

    Each sponsor gets one profile row in the sponsor_profiles table.
    That row is updated (never replaced) every time new information
    about the sponsor is discovered during an analysis run — a living
    document that grows richer with every analysis, never starting
    from scratch.

    Usage:
        store = SemanticStore()

        # Get what we know about a sponsor
        profile = await store.get_sponsor_profile("Novo Nordisk")

        # Update after analysing a study
        await store.update_sponsor_knowledge(
            sponsor="Novo Nordisk",
            results_posted=True,
            had_broken_promise=False,
            delay_days=5,
        )
    """

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        logger.info("SemanticStore initialised")

    async def _ensure_pool(self) -> None:
        """
        Creates the connection pool if it does not already exist.

        Called at the start of every public method. Follows the same
        lazy-initialisation pattern as EpisodicStore and
        ProceduralStore.
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
        )

        logger.info("SemanticStore pool created")

    @property
    def pool(self) -> asyncpg.Pool:
        """
        Returns the active connection pool, raising a clear error
        if the pool has not been created yet. Narrows the type from
        asyncpg.Pool | None to asyncpg.Pool for the type checker.
        """
        if self._pool is None:
            raise RuntimeError(
                "SemanticStore pool is not initialised — call "
                "await store._ensure_pool() first, or call a public "
                "method (get_sponsor_profile, update_sponsor_knowledge, "
                "etc.) which does this automatically."
            )
        return self._pool

    async def get_sponsor_profile(self, sponsor: str) -> dict[str, Any] | None:
        """
        Retrieves everything we know about a specific sponsor.

        Returns a dictionary containing the sponsor's full profile:
            sponsor           — the sponsor's name
            credibility_score — 0.0 (worst) to 1.0 (best)
            total_studies     — how many studies we have analysed
            results_posted    — how many times results were posted on time
            results_missing   — how many times results were NOT posted
            broken_promises   — how many outcome switches were detected
            avg_delay_days    — average days late on timeline
            last_updated      — when this profile was last modified

        The credibility score is not a simple average — it weights
        70% results compliance rate (posted / total studies) and 30%
        promise keeping (reduced per broken promise), clamped to a
        0.0–1.0 range. Below 0.6 triggers a LOW_CREDIBILITY signal.

        Args:
            sponsor: The sponsor name to look up. Must match exactly
                how the sponsor appears in the studies table.

        Returns:
            Dictionary with all profile fields, or None if no profile
            exists yet for this sponsor.
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                WHERE sponsor = $1
                """,
                sponsor,
            )

        if row is None:
            logger.info(f"No profile found for sponsor | sponsor={sponsor}")
            return None

        return {
            "sponsor": row["sponsor"],
            "credibility_score": float(row["credibility_score"] or 0.0),
            "total_studies": int(row["total_studies"] or 0),
            "results_posted": int(row["results_posted"] or 0),
            "results_missing": int(row["results_missing"] or 0),
            "broken_promises": int(row["broken_promises"] or 0),
            "avg_delay_days": float(row["avg_delay_days"] or 0.0),
            "last_updated": str(row["last_updated"]),
        }

    async def update_sponsor_knowledge(
        self,
        sponsor: str,
        results_posted: bool = False,
        had_broken_promise: bool = False,
        delay_days: int = 0,
    ) -> None:
        """
        Updates a sponsor's profile with new information from one study.

        Uses the UPSERT pattern (`INSERT ... ON CONFLICT ... DO
        UPDATE`) to create a new profile if the sponsor has never been
        seen before, or update the existing one otherwise. UPSERT is
        atomic and thread-safe, avoiding the race condition of a
        separate SELECT-then-INSERT-or-UPDATE approach when multiple
        agents update profiles in parallel.

        The average delay is maintained as a running mean using the
        incremental formula:
            new_avg = (old_avg * old_count + new_value) / (old_count + 1)

        After the counts are updated, credibility is recalculated in a
        second query (so it reflects the just-updated counts, not the
        pre-update values):
            compliance_rate = results_posted_count / total_studies
            promise_penalty = broken_promises * 0.1
            credibility = clamp(compliance_rate * 0.7 - promise_penalty, 0.0, 1.0)

        Args:
            sponsor: The sponsor name. Created if it doesn't yet exist.
            results_posted: Whether results were posted for this study.
            had_broken_promise: Whether outcome switching was detected.
            delay_days: How many days late this study was (0 = on time).
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sponsor_profiles (
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                )
                VALUES ($1, 0.5, 1, $2, $3, $4, $5, NOW())
                ON CONFLICT (sponsor) DO UPDATE SET
                    total_studies   = sponsor_profiles.total_studies + 1,
                    results_posted  = sponsor_profiles.results_posted + $2,
                    results_missing = sponsor_profiles.results_missing + $3,
                    broken_promises = sponsor_profiles.broken_promises + $4,
                    avg_delay_days  = (
                        (sponsor_profiles.avg_delay_days *
                         sponsor_profiles.total_studies) + $5
                    ) / (sponsor_profiles.total_studies + 1),
                    last_updated    = NOW()
                """,
                sponsor,
                int(results_posted),
                int(not results_posted),
                int(had_broken_promise),
                float(delay_days),
            )

            await conn.execute(
                """
                UPDATE sponsor_profiles
                SET credibility_score = GREATEST(0.0, LEAST(1.0,
                    (
                        CASE
                            WHEN total_studies = 0 THEN 0.5
                            ELSE (results_posted::float / total_studies) * 0.7
                        END
                    ) - (broken_promises * 0.1)
                ))
                WHERE sponsor = $1
                """,
                sponsor,
            )

        logger.info(
            f"Sponsor knowledge updated | "
            f"sponsor={sponsor} | "
            f"results_posted={results_posted} | "
            f"broken_promise={had_broken_promise} | "
            f"delay_days={delay_days}"
        )

    async def get_low_credibility_sponsors(
        self,
        threshold: float = 0.6,
        min_studies: int = 3,
    ) -> list[dict]:
        """
        Returns all sponsors whose credibility is below the threshold.

        Used by:
            1. The Track Record agent, to quickly identify problematic
               sponsors.
            2. The Pattern Finder agent, to check if a sponsor is a
               repeat offender.
            3. The API endpoint GET /api/v1/sponsors, for analyst
               dashboards.

        Args:
            threshold: Credibility below this score qualifies as "low".
            min_studies: Minimum number of studies required before a
                sponsor is flagged — a sponsor with only one study and
                one issue may just be unlucky, so we require enough
                data for a statistically meaningful judgment.

        Returns:
            List of sponsor profile dicts ordered by credibility
            ascending (worst sponsors first). Empty if none qualify.
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                WHERE credibility_score < $1
                  AND total_studies >= $2
                ORDER BY credibility_score ASC
                """,
                threshold,
                min_studies,
            )

        sponsors = [
            {
                "sponsor": row["sponsor"],
                "credibility_score": float(row["credibility_score"] or 0.0),
                "total_studies": int(row["total_studies"] or 0),
                "results_posted": int(row["results_posted"] or 0),
                "results_missing": int(row["results_missing"] or 0),
                "broken_promises": int(row["broken_promises"] or 0),
                "avg_delay_days": float(row["avg_delay_days"] or 0.0),
                "last_updated": str(row["last_updated"]),
            }
            for row in rows
        ]

        logger.info(
            f"Low credibility sponsors found | "
            f"count={len(sponsors)} | "
            f"threshold={threshold} | "
            f"min_studies={min_studies}"
        )

        return sponsors

    async def get_all_sponsor_profiles(self, limit: int = 50) -> list[dict]:
        """
        Returns all sponsor profiles ordered by credibility.

        Used by the API for analytics dashboards, showing analysts the
        full picture of every sponsor we have knowledge about.

        Args:
            limit: Maximum number of profiles to return.

        Returns:
            List of sponsor profiles, lowest credibility first.
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                ORDER BY credibility_score ASC
                LIMIT $1
                """,
                limit,
            )

        return [
            {
                "sponsor": row["sponsor"],
                "credibility_score": float(row["credibility_score"] or 0.0),
                "total_studies": int(row["total_studies"] or 0),
                "results_posted": int(row["results_posted"] or 0),
                "results_missing": int(row["results_missing"] or 0),
                "broken_promises": int(row["broken_promises"] or 0),
                "avg_delay_days": float(row["avg_delay_days"] or 0.0),
                "last_updated": str(row["last_updated"]),
            }
            for row in rows
        ]

    async def sponsor_exists(self, sponsor: str) -> bool:
        """
        Checks whether a sponsor profile already exists.

        Used before creating a new profile to avoid duplicate entries,
        and by agents to decide whether to load a profile or note
        that this sponsor has never been seen before.

        Args:
            sponsor: The sponsor name to check.

        Returns:
            True if a profile exists, False if this is a new sponsor.
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM sponsor_profiles WHERE sponsor = $1",
                sponsor,
            )

        return (count or 0) > 0

    async def close(self) -> None:
        """
        Closes the connection pool gracefully.

        Call this when the application shuts down.
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("SemanticStore pool closed")