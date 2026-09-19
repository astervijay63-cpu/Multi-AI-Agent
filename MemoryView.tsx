import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import {
  AGENTS,
  AGENT_LABELS,
  type EpisodesResponse,
  type ProceduresResponse,
} from "../types";
import { AgentTag, Empty, ErrorBox, Spinner, Stat, formatDate } from "./ui";

export function MemoryView() {
  const [tab, setTab] = useState<"episodes" | "procedures">("episodes");

  return (
    <div className="view">
      <header className="view-header">
        <div>
          <h2>Agent memory</h2>
          <p className="sub">
            What the agents remember doing (episodic) and the rules they reason
            by (procedural).
          </p>
        </div>
      </header>

      <div className="subtabs">
        <button
          className={`subtab ${tab === "episodes" ? "on" : ""}`}
          onClick={() => setTab("episodes")}
        >
          Episodes
        </button>
        <button
          className={`subtab ${tab === "procedures" ? "on" : ""}`}
          onClick={() => setTab("procedures")}
        >
          Procedures
        </button>
      </div>

      {tab === "episodes" ? <Episodes /> : <Procedures />}
    </div>
  );
}

function Episodes() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [agentName, setAgentName] = useState("");
  const [limit, setLimit] = useState(10);

  const [data, setData] = useState<EpisodesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await api.episodes({
          query: submitted || undefined,
          agent_name: agentName || undefined,
          limit,
        }),
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [submitted, agentName, limit]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <>
      <form
        className="filters panel"
        onSubmit={(e) => {
          e.preventDefault();
          // Searching embeds the query, so it fires on submit rather than
          // on every keystroke — each search is an embedding API call.
          setSubmitted(query.trim());
        }}
      >
        <label className="field grow">
          <span className="field-label">Search by meaning</span>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Leave empty for the most recent episodes"
          />
        </label>

        <label className="field">
          <span className="field-label">Agent</span>
          <select value={agentName} onChange={(e) => setAgentName(e.target.value)}>
            <option value="">All agents</option>
            {AGENTS.map((a) => (
              <option key={a} value={a}>
                {AGENT_LABELS[a]}
              </option>
            ))}
          </select>
        </label>

        <label className="field narrow">
          <span className="field-label">Limit</span>
          <input
            type="number"
            min={1}
            max={50}
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
          />
        </label>

        <button className="btn primary" type="submit">
          Search
        </button>
      </form>

      {error && <ErrorBox error={error} onRetry={() => void load()} />}
      {loading && <Spinner label="Loading episodes…" />}
      {!loading && !error && data?.episodes.length === 0 && (
        <Empty>No episodes recorded yet.</Empty>
      )}

      <div className="episode-list">
        {data?.episodes.map((episode) => (
          <article className="panel episode" key={episode.episode_id}>
            <header>
              <div className="tag-row">
                <AgentTag agent={episode.agent_name} />
                {episode.outcome && (
                  <span className={`tag outcome outcome-${episode.outcome}`}>
                    {episode.outcome.replace(/_/g, " ")}
                  </span>
                )}
                {episode.nct_id && (
                  <span className="mono nct">{episode.nct_id}</span>
                )}
              </div>
              <span className="muted small nowrap">
                {typeof episode.similarity === "number" && (
                  <span className="similarity">
                    {(episode.similarity * 100).toFixed(0)}% match ·{" "}
                  </span>
                )}
                {formatDate(episode.created_at)}
              </span>
            </header>
            <p>{episode.content}</p>
          </article>
        ))}
      </div>
    </>
  );
}

function Procedures() {
  const [agentName, setAgentName] = useState<string>(AGENTS[0]);
  const [data, setData] = useState<ProceduresResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.procedures(agentName));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [agentName]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <>
      <div className="filters panel">
        <label className="field grow">
          <span className="field-label">Agent</span>
          <select value={agentName} onChange={(e) => setAgentName(e.target.value)}>
            {AGENTS.map((a) => (
              <option key={a} value={a}>
                {AGENT_LABELS[a]}
              </option>
            ))}
          </select>
        </label>

        {data && (
          <div className="stat-row inline">
            <Stat label="Total rules" value={data.total_rules} />
            <Stat label="Default" value={data.default_rules} />
            <Stat
              label="Learned"
              value={data.learned_rules}
              hint="from human rejections"
            />
          </div>
        )}
      </div>

      {error && <ErrorBox error={error} onRetry={() => void load()} />}
      {loading && <Spinner label="Loading rules…" />}
      {!loading && !error && data?.procedures.length === 0 && (
        <Empty>This agent has no rules yet.</Empty>
      )}

      <div className="rule-list">
        {data?.procedures.map((rule) => (
          <div
            className={`panel rule rule-${rule.rule_type}`}
            key={rule.procedure_id}
          >
            <div className="rule-body">
              <span className={`tag rule-type rule-type-${rule.rule_type}`}>
                {rule.rule_type}
              </span>
              <p>{rule.rule_text}</p>
            </div>
            <div className="muted small nowrap">{formatDate(rule.created_at)}</div>
          </div>
        ))}
      </div>
    </>
  );
}
