"""
Procedural memory store for MOSAIC agents.

Procedural memory encodes HOW an agent should reason, as opposed to
episodic memory, which encodes WHAT happened in past sessions. Each
agent starts with a set of DEFAULT reasoning rules. When a human
reviewer rejects a signal and explains why, that rejection reason is
written into the agent's procedures table as a new LEARNED rule,
permanently changing how the agent reasons going forward.
"""
import asyncpg
import json
from datetime import datetime

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


DEFAULT_RULES = {
    "missing_results_agent": [
        "Flag a study as missing results ONLY if status is COMPLETED "
        "and results_posted is False and more than 12 months have "
        "passed since the completion date.",

        "Do NOT flag studies with status TERMINATED as missing results. "
        "Terminated trials are not legally required to post results "
        "in all circumstances — termination often means the study "
        "was stopped early and has incomplete data.",

        "If enrollment was zero or very low (under 10 participants), "
        "note this in the signal but reduce confidence to 0.5. "
        "A study that never really started may not have reportable results.",

        "Always check the sponsor's track record before assigning "
        "a confidence score. A first-time missing result from a "
        "historically compliant sponsor warrants lower confidence "
        "than the same finding from a repeat offender.",
    ],

    "broken_promises_agent": [
        "Flag outcome switching ONLY when the PRIMARY outcome changes "
        "after enrollment has begun. Changes to secondary outcomes "
        "are less concerning and should not trigger a HIGH confidence signal.",

        "A change in outcome MEASUREMENT METHOD (how it is measured) "
        "is different from a change in the outcome itself. "
        "Method changes may be legitimate protocol improvements — "
        "flag them at MEDIUM confidence, not HIGH.",

        "If a protocol amendment was filed BEFORE enrollment began, "
        "the outcome change is less suspicious — the study had not "
        "yet collected data that could have influenced the change. "
        "Assign MEDIUM confidence in this case.",

        "Always note the date of the change relative to the "
        "enrollment start date — this timing is the most important "
        "factor in assessing whether outcome switching is intentional.",
    ],

    "track_record_agent": [
        "A credibility score below 0.6 should trigger a LOW_CREDIBILITY "
        "signal. Between 0.6 and 0.75 is concerning but not alarming — "
        "note it in the analysis but do not generate a signal.",

        "Weight recent behaviour more heavily than old behaviour. "
        "A sponsor with 5 violations in the last 2 years is more "
        "concerning than one with 10 violations spread over 20 years.",

        "If a sponsor has fewer than 3 studies in our database, "
        "reduce confidence to 0.5. We do not have enough data to "
        "make a reliable judgment about their track record.",

        "Always distinguish between a sponsor's PRIMARY studies "
        "(where they are the lead sponsor) and COLLABORATIVE studies "
        "(where they are a secondary party). Hold them more accountable "
        "for their primary studies.",
    ],

    "pattern_finder_agent": [
        "A cross-study pattern requires at least 3 studies to be "
        "meaningful. Two studies with similar issues may be coincidence. "
        "Three or more is a pattern worth flagging.",

        "When multiple companies are testing the same drug for the "
        "same condition, check whether any of them have hidden "
        "negative results from previous studies in our database.",

        "A drug that failed Phase 2 for condition A but is being "
        "retried in Phase 2 for condition B is worth flagging — "
        "especially if the mechanism of action is the same.",

        "Patterns across the same SPONSOR are more actionable than "
        "patterns across different sponsors. Same-sponsor patterns "
        "suggest systemic issues, not coincidence.",
    ],

    "side_effect_agent": [
        "A safety discrepancy between the official filing and a "
        "published paper is only meaningful if the paper was published "
        "AFTER the trial completed — not during it.",

        "Look specifically for cases where the filing says "
        "'no serious adverse events' but published papers mention "
        "hospitalisations, discontinuations, or deaths. "
        "This is the highest-priority safety signal.",

        "If the discrepancy is minor (e.g. different terminology "
        "for the same event), assign LOW confidence. "
        "If the discrepancy involves severity (mild vs serious), "
        "assign HIGH confidence.",

        "Always note whether the paper's authors are the same as "
        "the trial's investigators. Independent authors are more "
        "credible than sponsor-employed investigators.",
    ],

    "timeline_agent": [
        "Flag a delay ONLY if it exceeds 180 days beyond the "
        "stated completion date AND no amendment was filed explaining "
        "the extension. A silent delay is more suspicious than "
        "a disclosed one.",

        "COVID-19 is a legitimate reason for delays between "
        "March 2020 and December 2022. Do not flag delays in this "
        "period as suspicious without additional evidence.",

        "A study that is recruiting past its stated completion date "
        "may simply have underestimated enrollment time — this is "
        "common and not inherently suspicious. Focus on COMPLETED "
        "studies that are past their results posting deadline.",

        "Always compare the actual completion date against BOTH "
        "the original completion date AND any amended completion "
        "dates. Use the most recent amendment as the baseline.",
    ],
}


class ProceduralStore:
    """
    Stores and retrieves agent reasoning rules (procedures).

    Each agent has its own set of procedures — rules that guide how
    it reasons about clinical trial data. Procedures come in two
    types:

    1. DEFAULT — built-in rules the agent always starts with.
    2. LEARNED — rules added when a human reviewer corrects the agent.

    Usage:
        store = ProceduralStore()

        # Load all rules for an agent before it starts reasoning
        rules = await store.get_procedures("missing_results_agent")

        # Update rules after a human rejection
        await store.update_from_feedback(
            agent_name="missing_results_agent",
            rejection_reason="Terminated trials should not be flagged"
        )
    """

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        logger.info("ProceduralStore initialised")

    async def _ensure_pool(self) -> None:
        """
        Creates the database connection pool if it does not already exist.

        Called at the start of every public method to guarantee the
        database is reachable before use.
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
            max_size=3,
        )

        logger.info("ProceduralStore pool created")

    @property
    def pool(self) -> asyncpg.Pool:
        """
        Returns the active connection pool, raising a clear error
        if the pool has not been created yet. Narrows the type from
        asyncpg.Pool | None to asyncpg.Pool for the type checker.
        """
        if self._pool is None:
            raise RuntimeError(
                "ProceduralStore pool is not initialised — call "
                "await store._ensure_pool() first, or call a public "
                "method (get_procedures, update_from_feedback, etc.) "
                "which does this automatically."
            )
        return self._pool

    async def initialise_defaults(self) -> None:
        """
        Inserts the DEFAULT_RULES into the procedures table.

        Called once, when MOSAIC starts up for the very first time.
        Safe to call multiple times — existing rows are skipped via
        `ON CONFLICT DO NOTHING`, so re-running this will not
        duplicate any rules.

        Storing rules in the database (rather than hardcoding them in
        the agent) keeps them persistent across restarts, updateable
        by humans, visible via the API, and auditable over time.
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            for agent_name, rules in DEFAULT_RULES.items():
                for rule_text in rules:
                    # WHY NOT "ON CONFLICT DO NOTHING":
                    # That clause needs something to conflict WITH. There is
                    # no unique index on (agent_name, rule_text), so nothing
                    # ever conflicted and every startup inserted a fresh copy
                    # of all 24 default rules. Agents were loading 128 rules
                    # apiece — the same 4 repeated 32 times — which bloated
                    # every system prompt and grew on each restart.
                    # WHERE NOT EXISTS makes the insert genuinely idempotent
                    # without requiring a schema migration.
                    await conn.execute(
                        """
                        INSERT INTO procedures (
                            agent_name,
                            rule_text,
                            rule_type,
                            source
                        )
                        SELECT $1, $2, $3, $4
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM procedures
                            WHERE agent_name = $1
                              AND rule_text  = $2
                        )
                        """,
                        agent_name,
                        rule_text,
                        "default",
                        "default",
                    )

        logger.info(
            f"Default procedures initialised | "
            f"agents={list(DEFAULT_RULES.keys())}"
        )

    async def get_procedures(self, agent_name: str) -> list[str]:
        """
        Returns all reasoning rules for a specific agent.

        Called at the start of every agent session, before any
        analysis. The agent reads these rules and incorporates them
        into its system prompt so that its reasoning follows them.

        The returned list contains both default rules (built-in at
        startup) and learned rules (added from human feedback over
        time). Rules are returned oldest first, so defaults precede
        learned corrections in the order they were added. The agent
        treats all rules as equally authoritative — it does not
        distinguish default from learned.

        Args:
            agent_name: Which agent's rules to retrieve.

        Returns:
            List of rule strings, ordered oldest first. Empty list if
            no rules are found (should not happen in practice).
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT rule_text
                FROM procedures
                WHERE agent_name = $1
                ORDER BY created_at ASC
                """,
                agent_name,
            )

        rules = [row["rule_text"] for row in rows]

        logger.info(
            f"Procedures loaded | "
            f"agent={agent_name} | "
            f"rules_count={len(rules)}"
        )

        return rules

    async def update_from_feedback(
        self,
        agent_name: str,
        rejection_reason: str,
    ) -> str:
        """
        Adds a new learned reasoning rule from a human rejection.

        This is the core of the learning loop. When a human reviewer
        rejects an agent's signal, they explain why it was wrong; that
        explanation is stored verbatim as a new rule in the procedures
        table. From this point forward, every run of this agent loads
        the rule via `get_procedures` and applies it, avoiding a
        repeat of the same mistake.

        Args:
            agent_name: Which agent to add the rule to.
            rejection_reason: The human's explanation of what was
                wrong. Stored as-is as the new rule's text.

        Returns:
            The procedure_id (UUID string) of the newly created rule.
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            procedure_id = await conn.fetchval(
                """
                INSERT INTO procedures (
                    agent_name,
                    rule_text,
                    rule_type,
                    source
                )
                VALUES ($1, $2, $3, $4)
                RETURNING procedure_id
                """,
                agent_name,
                rejection_reason,
                "learned",
                "hitl_rejection",
            )

        logger.info(
            f"Procedure learned from feedback | "
            f"agent={agent_name} | "
            f"rule_preview='{rejection_reason[:80]}...' | "
            f"procedure_id={procedure_id}"
        )

        return str(procedure_id)

    async def get_all_procedures_for_api(self, agent_name: str) -> list[dict]:
        """
        Returns all procedures for an agent with full metadata.

        Unlike `get_procedures`, which returns only rule text, this
        returns the full procedure record — including rule type,
        source, and timestamps — for use by:

            GET /api/v1/memory/procedures/{agent_name}

        This lets analysts see what rules an agent currently follows,
        which are built-in vs. learned from feedback, and when each
        rule was added.

        Args:
            agent_name: Which agent's procedures to return.

        Returns:
            List of procedure dictionaries with full metadata.
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    procedure_id,
                    agent_name,
                    rule_text,
                    rule_type,
                    source,
                    created_at
                FROM procedures
                WHERE agent_name = $1
                ORDER BY created_at ASC
                """,
                agent_name,
            )

        return [
            {
                "procedure_id": str(row["procedure_id"]),
                "agent_name": row["agent_name"],
                "rule_text": row["rule_text"],
                "rule_type": row["rule_type"],
                "source": row["source"],
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    async def close(self) -> None:
        """
        Closes the connection pool gracefully.

        Call this when the application shuts down.
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("ProceduralStore pool closed")