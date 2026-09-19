import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import MosaicState, SignalOutput
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


llm = ChatGoogleGenerativeAI(
    model=settings.google_chat_model,
    temperature=0.1,
    api_key=settings.google_api_key,
)


async def supervisor_route(state: MosaicState) -> dict:
    """
    The entry point of every MOSAIC analysis run.

    WHAT THIS FUNCTION DOES:
    1. Generates a unique run ID for this analysis session
    2. Logs what task is being investigated
    3. Returns an updated state that all specialists will receive

    WHY SO SIMPLE?
    The supervisor does NOT need to read the task and decide which
    specialists to activate — we always run ALL six specialists in
    parallel for every task. This is by design:
    - Different agents may find different signals in the same task
    - Running all six costs the same time as running one (parallel)
    - We never miss a signal type by selectively routing

    LANGGRAPH NODE CONTRACT:
    Every LangGraph node must:
    - Accept: the current MosaicState
    - Return: a dict of ONLY the fields that changed
    LangGraph automatically merges the returned dict into the full state.
    You do not return the entire state — just your changes.

    Args:
        state: The current MosaicState from LangGraph.

    Returns:
        Dict with updated run_id, agents_activated, and signals fields.
    """

    run_id = str(uuid.uuid4())

    logger.info(
        f"Supervisor routing | "
        f"run_id={run_id} | "
        f"task='{state.get('task', '')[:80]}'"
    )

    return {
        "run_id":           run_id,
        "agents_activated": [],
        "signals":          [],
        "run_complete":     False,
        "error_log":        [],
    }


async def supervisor_compile(state: MosaicState) -> dict:
    """
    Reads all agent signals and compiles the final intelligence brief.

    WHEN THIS RUNS:
    LangGraph calls this node only after ALL six specialist nodes
    have completed. This is guaranteed by the graph structure in
    graph_builder.py — all specialists connect to this node.

    WHAT THIS FUNCTION DOES:
    1. Collects all signals from state (from all 6 agents)
    2. Separates high-confidence signals from those needing review
    3. Uses GPT-4o to write a professional intelligence brief
    4. Returns the completed state

    WHY USE GPT-4o TO WRITE THE BRIEF?
    The raw signals are structured data — JSON with fields like
    summary, confidence, nct_id. They are accurate but not readable.
    GPT-4o transforms them into a professional narrative brief that
    a human analyst can read and act on immediately.
    The signals provide the FACTS. GPT-4o provides the WRITING.

    Args:
        state: The full MosaicState — now populated with all agent signals.

    Returns:
        Dict with final_brief, run_complete=True, and summary stats.
    """

    signals          = state.get("signals", [])
    agents_activated = state.get("agents_activated", [])
    task             = state.get("task", "")

    logger.info(
        f"Supervisor compiling brief | "
        f"signals={len(signals)} | "
        f"agents_activated={len(agents_activated)}"
    )

    if not signals:
        logger.info("No signals found — returning clean brief")
        return {
            "final_brief":  (
                "**EXECUTIVE SUMMARY:** Analysis complete. "
                "No significant research integrity signals were detected "
                "for the specified task and study set."
            ),
            "run_complete":     True,
        }

    high_confidence_signals = [
        s for s in signals
        if s.get("confidence", 0) >= 0.6
    ]

    review_signals = [
        s for s in signals
        if s.get("confidence", 0) < 0.6
    ]

    signals_text = _format_signals_for_llm(signals)

    system_prompt = """You are the Chief Intelligence Officer of MOSAIC —
                    a clinical trial research integrity system. Your job is to compile
                    a professional executive intelligence brief from the signals generated
                    by specialist AI agents.

                    BRIEF FORMAT:
                    1. EXECUTIVE SUMMARY — 2-3 sentences summarising the most critical findings
                    2. SIGNALS BY PRIORITY — each signal as a numbered item with:
                    - What was found
                    - Why it matters
                    - What action to take
                    3. SIGNALS REQUIRING HUMAN REVIEW — list any low-confidence signals
                    4. PIPELINE HEALTH — note any errors or issues during the run

                    TONE: Professional, factual, actionable. Write as if briefing a
                    senior compliance officer or investigative journalist.
                    Be specific — include NCT IDs, sponsor names, and exact timeframes.
                    """

    human_prompt = f"""
            ANALYSIS TASK: {task}

            SIGNALS FOUND BY AGENTS:
            {signals_text}

            HIGH CONFIDENCE SIGNALS: {len(high_confidence_signals)}
            SIGNALS REQUIRING REVIEW: {len(review_signals)}
            AGENTS ACTIVATED: {', '.join(agents_activated)}

            Please compile the final intelligence brief now.
        """

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )

        final_brief = response.content

        logger.info(
            f"Brief compiled successfully | "
            f"signals_included={len(signals)} | "
            f"brief_length={len(final_brief)} chars"
        )

    except Exception as e:
        logger.error(f"LLM brief compilation failed | error={e}")
        final_brief = _fallback_brief(signals, agents_activated, task)

    # NOTE: do NOT return "agents_activated" here.
    # It is an append-only channel (see _merge_lists in graph/state.py) —
    # the six specialists have already written all six names into it by
    # the time this node runs. Returning the list we just READ would make
    # the reducer append it a second time, doubling it to twelve entries.
    return {
        "final_brief":              final_brief,
        "run_complete":             True,
        "total_signals":            len(signals),
        "signals_requiring_review": len(review_signals),
    }


def _format_signals_for_llm(signals: list[SignalOutput]) -> str:
    """
    Converts a list of signal dicts into a clean, readable text block
    that GPT-4o can effectively summarise into the final brief.

    WHY FORMAT BEFORE SENDING TO GPT-4o?
    Raw signal dicts are JSON — full of curly braces and quotes.
    GPT-4o works better with plain, labelled text than raw JSON.
    Formatting the signals into clear sections produces better briefs.

    Args:
        signals: List of SignalOutput dicts from specialist agents.

    Returns:
        A formatted string with all signals clearly laid out.
    """

    if not signals:
        return "No signals generated."

    lines = []

    for i, signal in enumerate(signals, start=1):
        lines.append(f"SIGNAL {i}:")
        lines.append(f"  Agent:       {signal.get('agent', 'unknown')}")
        lines.append(f"  Type:        {signal.get('signal_type', 'unknown')}")
        lines.append(f"  NCT ID:      {signal.get('nct_id', 'N/A')}")
        lines.append(f"  Confidence:  {signal.get('confidence', 0.0):.2f}")
        lines.append(f"  Summary:     {signal.get('summary', '')}")
        lines.append("")

    return "\n".join(lines)


def _fallback_brief(
    signals:          list,
    agents_activated: list,
    task:             str,
) -> str:
    """
    Generates a basic structured brief WITHOUT using GPT-4o.

    Called when the LLM call fails — ensures the API always returns
    something useful even if OpenAI is down or rate-limited.
    The output is less polished than the GPT-4o brief but contains
    all the factual information the caller needs.

    Args:
        signals:          All signals from the run.
        agents_activated: Which agents ran.
        task:             The original analysis task.

    Returns:
        A plain text brief built directly from signal data.
    """

    lines = [
        "**EXECUTIVE SUMMARY:**",
        f"Analysis complete. {len(signals)} signal(s) detected.",
        "",
        "**SIGNALS BY PRIORITY:**",
        "",
    ]

    for i, signal in enumerate(signals, start=1):
        lines.append(
            f"{i}. **{signal.get('nct_id', 'Unknown')} "
            f"- {signal.get('signal_type', 'Unknown')}:**"
        )
        lines.append(f"   {signal.get('summary', 'No summary available.')}")
        lines.append(
            f"   Confidence: {signal.get('confidence', 0.0):.2f} | "
            f"Agent: {signal.get('agent', 'unknown')}"
        )
        lines.append("")

    lines.append(f"**AGENTS ACTIVATED:** {', '.join(agents_activated)}")
    lines.append(
        "\n*Note: This brief was generated without LLM assistance "
        "due to a temporary error. Please review raw signals directly.*"
    )

    return "\n".join(lines)
