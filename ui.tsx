import type { ReactNode } from "react";
import { AGENT_LABELS, SIGNAL_TYPE_LABELS } from "../types";

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="spinner-row">
      <span className="spinner" aria-hidden="true" />
      <span>{label ?? "Loading…"}</span>
    </div>
  );
}

export function ErrorBox({
  error,
  onRetry,
}: {
  error: string;
  onRetry?: () => void;
}) {
  return (
    <div className="error-box" role="alert">
      <strong>Something went wrong</strong>
      <p>{error}</p>
      {onRetry && (
        <button className="btn ghost" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function AgentTag({ agent }: { agent: string }) {
  return (
    <span className={`tag agent agent-${agent}`}>
      {AGENT_LABELS[agent] ?? agent}
    </span>
  );
}

export function TypeTag({ type }: { type: string }) {
  return <span className="tag type">{SIGNAL_TYPE_LABELS[type] ?? type}</span>;
}

export function StatusTag({ status }: { status: string }) {
  return <span className={`tag status status-${status}`}>{status}</span>;
}

// Confidence drives the HITL gate: anything under 0.6 goes to a human
// (see supervisor_compile), so the bar is colour-banded at that line
// to make "this one needs review" visible without reading the number.
export function Confidence({ value }: { value: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const band = value >= 0.8 ? "high" : value >= 0.6 ? "mid" : "low";
  return (
    <span className="confidence" title={`Confidence ${value.toFixed(2)}`}>
      <span className="confidence-bar">
        <span className={`confidence-fill ${band}`} style={{ width: `${pct}%` }} />
      </span>
      <span className="confidence-num">{value.toFixed(2)}</span>
    </span>
  );
}

export function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
}) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value.replace(" ", "T"));
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

// Evidence comes back as a JSON-encoded array from asyncpg in some
// paths and a real array in others; normalise both to a string list.
export function toEvidenceList(evidence: unknown): string[] {
  if (!evidence) return [];
  if (Array.isArray(evidence)) return evidence.map(String);
  if (typeof evidence === "string") {
    try {
      const parsed = JSON.parse(evidence);
      if (Array.isArray(parsed)) return parsed.map(String);
    } catch {
      // Not JSON — treat the whole string as a single piece of evidence.
    }
    return [evidence];
  }
  return [];
}
