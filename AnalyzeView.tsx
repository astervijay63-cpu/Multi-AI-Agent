import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";
import { AGENTS, AGENT_LABELS, type AnalysisResponse } from "../types";
import { Brief } from "./Brief";
import { ErrorBox, Stat } from "./ui";

const EXAMPLE_TASKS = [
  "Find completed trials where sponsor never posted results",
  "Identify sponsors with a pattern of unexplained timeline delays",
  "Check for trials with reported adverse events missing from publications",
];

export function AnalyzeView({ onRunComplete }: { onRunComplete: () => void }) {
  const [task, setTask] = useState(EXAMPLE_TASKS[0]);
  const [nctInput, setNctInput] = useState("");
  const [maxStudies, setMaxStudies] = useState(10);

  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  // A run takes minutes, and the endpoint is not streaming — there is no
  // progress to report until it returns. An elapsed counter is the honest
  // signal that work is still happening, rather than a fake progress bar.
  useEffect(() => {
    if (!running) return;
    const started = Date.now();
    setElapsed(0);
    const id = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - started) / 1000)),
      1000,
    );
    return () => window.clearInterval(id);
  }, [running]);

  // Abort the in-flight run if the view unmounts, so a navigating user
  // does not leave a fetch hanging with nowhere to deliver its result.
  useEffect(() => () => abortRef.current?.abort(), []);

  const parseNctIds = (raw: string): string[] =>
    raw
      .split(/[\s,]+/)
      .map((s) => s.trim().toUpperCase())
      .filter(Boolean);

  async function runAnalysis(event: React.FormEvent) {
    event.preventDefault();
    if (running) return;

    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.analyze(
        {
          task: task.trim(),
          nct_ids: parseNctIds(nctInput),
          max_studies: maxStudies,
        },
        controller.signal,
      );
      setResult(response);
      // New signals are now in the database, so the header counters and
      // the review queue are stale until the parent refreshes them.
      onRunComplete();
    } catch (err) {
      if (controller.signal.aborted) {
        setError("Run cancelled.");
      } else {
        setError(err instanceof ApiError ? err.message : String(err));
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }

  const activated = new Set(result?.agents_activated ?? []);

  return (
    <div className="view">
      <header className="view-header">
        <div>
          <h2>Run an analysis</h2>
          <p className="sub">
            All six specialists run in parallel against the study corpus, then
            the supervisor compiles their signals into one brief.
          </p>
        </div>
      </header>

      <form className="panel analyze-form" onSubmit={runAnalysis}>
        <label className="field">
          <span className="field-label">Analysis task</span>
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            rows={3}
            required
            disabled={running}
            placeholder="Describe in plain English what the agents should investigate"
          />
        </label>

        <div className="chips">
          {EXAMPLE_TASKS.map((example) => (
            <button
              type="button"
              key={example}
              className="chip"
              disabled={running}
              onClick={() => setTask(example)}
            >
              {example}
            </button>
          ))}
        </div>

        <div className="field-row">
          <label className="field">
            <span className="field-label">
              NCT IDs <span className="muted">(optional)</span>
            </span>
            <input
              type="text"
              value={nctInput}
              onChange={(e) => setNctInput(e.target.value)}
              disabled={running}
              placeholder="NCT04788680, NCT02208921 — leave empty to search broadly"
            />
          </label>

          <label className="field narrow">
            <span className="field-label">Max studies</span>
            <input
              type="number"
              min={1}
              max={100}
              value={maxStudies}
              disabled={running}
              onChange={(e) => setMaxStudies(Number(e.target.value))}
            />
          </label>
        </div>

        <div className="form-actions">
          <button type="submit" className="btn primary" disabled={running}>
            {running ? "Analysis running…" : "Run analysis"}
          </button>
          {running && (
            <>
              <span className="elapsed">
                <span className="spinner" aria-hidden="true" />
                {formatElapsed(elapsed)} elapsed
              </span>
              <button
                type="button"
                className="btn ghost"
                onClick={() => abortRef.current?.abort()}
              >
                Cancel
              </button>
            </>
          )}
        </div>

        {running && (
          <p className="note">
            Six agents are reasoning and calling tools concurrently. Runs
            typically take one to three minutes. Cancelling stops this page
            waiting, but the backend run continues to completion.
          </p>
        )}
      </form>

      {error && <ErrorBox error={error} />}

      {result && (
        <section className="panel result">
          <div className="stat-row">
            <Stat label="Signals found" value={result.total_signals} />
            <Stat
              label="Needing review"
              value={result.signals_requiring_review}
              hint="confidence below 0.6"
            />
            <Stat label="Agents activated" value={result.agents_activated.length} />
            <Stat
              label="Duration"
              value={`${result.duration_seconds.toFixed(1)}s`}
            />
          </div>

          <div className="agent-grid">
            {AGENTS.map((agent) => (
              <div
                key={agent}
                className={`agent-pill ${activated.has(agent) ? "on" : "off"}`}
              >
                <span className="dot" />
                {AGENT_LABELS[agent]}
              </div>
            ))}
          </div>

          <h3 className="brief-heading">Intelligence brief</h3>
          <Brief text={result.final_brief} />

          <footer className="run-meta">
            Run <code>{result.run_id}</code>
          </footer>
        </section>
      )}
    </div>
  );
}

function formatElapsed(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return mins ? `${mins}m ${String(secs).padStart(2, "0")}s` : `${secs}s`;
}
