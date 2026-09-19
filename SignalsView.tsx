import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import {
  AGENTS,
  AGENT_LABELS,
  SIGNAL_TYPES,
  SIGNAL_TYPE_LABELS,
  type Signal,
} from "../types";
import {
  AgentTag,
  Confidence,
  Empty,
  ErrorBox,
  Spinner,
  StatusTag,
  TypeTag,
  formatDate,
  toEvidenceList,
} from "./ui";

export function SignalsView() {
  const [agent, setAgent] = useState("");
  const [signalType, setSignalType] = useState("");
  const [status, setStatus] = useState("");
  const [limit, setLimit] = useState(50);

  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Signal | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSignals(
        await api.listSignals({
          agent: agent || undefined,
          signal_type: signalType || undefined,
          status: status || undefined,
          limit,
        }),
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [agent, signalType, status, limit]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="view">
      <header className="view-header">
        <div>
          <h2>Signals</h2>
          <p className="sub">
            Every signal the agents have written to the database, newest first.
          </p>
        </div>
        <button className="btn ghost" onClick={() => void load()}>
          Refresh
        </button>
      </header>

      <div className="filters panel">
        <label className="field">
          <span className="field-label">Agent</span>
          <select value={agent} onChange={(e) => setAgent(e.target.value)}>
            <option value="">All agents</option>
            {AGENTS.map((a) => (
              <option key={a} value={a}>
                {AGENT_LABELS[a]}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field-label">Signal type</span>
          <select
            value={signalType}
            onChange={(e) => setSignalType(e.target.value)}
          >
            <option value="">All types</option>
            {SIGNAL_TYPES.map((t) => (
              <option key={t} value={t}>
                {SIGNAL_TYPE_LABELS[t]}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field-label">Status</span>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">All statuses</option>
            <option value="approved">Approved</option>
            <option value="pending">Pending</option>
            <option value="rejected">Rejected</option>
          </select>
        </label>

        <label className="field narrow">
          <span className="field-label">Limit</span>
          <input
            type="number"
            min={1}
            max={200}
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
          />
        </label>
      </div>

      {error && <ErrorBox error={error} onRetry={() => void load()} />}
      {loading && <Spinner label="Loading signals…" />}

      {!loading && !error && signals.length === 0 && (
        <Empty>
          No signals match these filters. Run an analysis to generate some.
        </Empty>
      )}

      {!loading && signals.length > 0 && (
        <div className="table-wrap panel">
          <table>
            <thead>
              <tr>
                <th>NCT ID</th>
                <th>Agent</th>
                <th>Type</th>
                <th className="wide">Summary</th>
                <th>Confidence</th>
                <th>Status</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((signal) => (
                <tr
                  key={signal.signal_id}
                  className="clickable"
                  onClick={() => setSelected(signal)}
                >
                  <td className="mono">{signal.nct_id || "—"}</td>
                  <td>
                    <AgentTag agent={signal.agent} />
                  </td>
                  <td>
                    <TypeTag type={signal.signal_type} />
                  </td>
                  <td className="wide">
                    <span className="truncate">{signal.summary}</span>
                  </td>
                  <td>
                    <Confidence value={signal.confidence} />
                  </td>
                  <td>
                    <StatusTag status={signal.status} />
                  </td>
                  <td className="muted nowrap">{formatDate(signal.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selected && (
        <SignalDetail
          signalId={selected.signal_id}
          fallback={selected}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

// The list query does not select `evidence`, so opening a signal refetches
// it by id to get the full record rather than showing a partial one.
function SignalDetail({
  signalId,
  fallback,
  onClose,
}: {
  signalId: string;
  fallback: Signal;
  onClose: () => void;
}) {
  const [signal, setSignal] = useState<Signal>(fallback);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .getSignal(signalId)
      .then((full) => {
        if (!cancelled) setSignal(full);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [signalId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const evidence = toEvidenceList(signal.evidence);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <header className="drawer-header">
          <div>
            <h3>{signal.nct_id || "No NCT ID"}</h3>
            <div className="tag-row">
              <AgentTag agent={signal.agent} />
              <TypeTag type={signal.signal_type} />
              <StatusTag status={signal.status} />
            </div>
          </div>
          <button className="btn ghost icon" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <dl className="detail-list">
          <dt>Confidence</dt>
          <dd>
            <Confidence value={signal.confidence} />
          </dd>
          <dt>Created</dt>
          <dd>{formatDate(signal.created_at)}</dd>
          <dt>Signal ID</dt>
          <dd className="mono small">{signal.signal_id}</dd>
        </dl>

        <h4>Summary</h4>
        <p>{signal.summary}</p>

        <h4>Evidence</h4>
        {loading && <Spinner label="Loading evidence…" />}
        {error && <ErrorBox error={error} />}
        {!loading && !error && evidence.length === 0 && (
          <Empty>No evidence was recorded for this signal.</Empty>
        )}
        {evidence.length > 0 && (
          <ul className="evidence">
            {evidence.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        )}
      </aside>
    </div>
  );
}
