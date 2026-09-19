import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { ReviewQueueItem, ReviewQueueResponse } from "../types";
import {
  AgentTag,
  Confidence,
  Empty,
  ErrorBox,
  Spinner,
  Stat,
  TypeTag,
} from "./ui";

type Decision = "approve" | "reject" | "edit";

export function ReviewView({ onDecision }: { onDecision: () => void }) {
  const [data, setData] = useState<ReviewQueueResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reviewer, setReviewer] = useState(
    () => localStorage.getItem("mosaic.reviewer") ?? "analyst",
  );
  const [flash, setFlash] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.reviewQueue());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    localStorage.setItem("mosaic.reviewer", reviewer);
  }, [reviewer]);

  async function submit(item: ReviewQueueItem, body: {
    decision: Decision;
    rejection_reason: string;
    edit_summary: string;
  }) {
    const response = await api.submitReview(item.review_id, {
      ...body,
      reviewer: reviewer.trim() || "analyst",
    });
    setFlash(response.message);
    await load();
    onDecision();
  }

  return (
    <div className="view">
      <header className="view-header">
        <div>
          <h2>Human review queue</h2>
          <p className="sub">
            Signals the agents were not confident enough to stand behind, most
            uncertain first. Rejecting one writes the reason into procedural
            memory — that agent reasons differently from then on.
          </p>
        </div>
        <button className="btn ghost" onClick={() => void load()}>
          Refresh
        </button>
      </header>

      {data && (
        <div className="stat-row panel">
          <Stat label="Pending" value={data.total_pending} />
          <Stat label="Approved" value={data.total_approved} />
          <Stat label="Rejected" value={data.total_rejected} />
          <label className="field reviewer">
            <span className="field-label">Reviewing as</span>
            <input
              type="text"
              value={reviewer}
              onChange={(e) => setReviewer(e.target.value)}
              placeholder="your name or email"
            />
          </label>
        </div>
      )}

      {flash && (
        <div className="flash" role="status">
          {flash}
          <button className="btn ghost icon" onClick={() => setFlash(null)}>
            ✕
          </button>
        </div>
      )}

      {error && <ErrorBox error={error} onRetry={() => void load()} />}
      {loading && <Spinner label="Loading review queue…" />}

      {!loading && !error && data?.queue.length === 0 && (
        <Empty>
          The queue is empty. Every signal so far cleared the confidence
          threshold or has already been reviewed.
        </Empty>
      )}

      <div className="review-list">
        {data?.queue.map((item) => (
          <ReviewCard key={item.review_id} item={item} onSubmit={submit} />
        ))}
      </div>
    </div>
  );
}

function ReviewCard({
  item,
  onSubmit,
}: {
  item: ReviewQueueItem;
  onSubmit: (
    item: ReviewQueueItem,
    body: { decision: Decision; rejection_reason: string; edit_summary: string },
  ) => Promise<void>;
}) {
  const [mode, setMode] = useState<Decision | null>(null);
  const [reason, setReason] = useState("");
  const [editSummary, setEditSummary] = useState(item.summary);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function commit(decision: Decision) {
    setBusy(true);
    setError(null);
    try {
      await onSubmit(item, {
        decision,
        rejection_reason: decision === "reject" ? reason.trim() : "",
        edit_summary: decision === "edit" ? editSummary.trim() : "",
      });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
      setBusy(false);
    }
  }

  return (
    <article className="panel review-card">
      <header>
        <div className="tag-row">
          <span className="mono nct">{item.nct_id || "No NCT ID"}</span>
          <AgentTag agent={item.agent} />
          <TypeTag type={item.signal_type} />
        </div>
        <Confidence value={item.confidence} />
      </header>

      <p className="review-summary">{item.summary}</p>

      {mode === null && (
        <div className="form-actions">
          <button
            className="btn approve"
            disabled={busy}
            onClick={() => void commit("approve")}
          >
            Approve
          </button>
          <button className="btn reject" disabled={busy} onClick={() => setMode("reject")}>
            Reject
          </button>
          <button className="btn ghost" disabled={busy} onClick={() => setMode("edit")}>
            Edit summary
          </button>
        </div>
      )}

      {mode === "reject" && (
        <div className="decision-form">
          <label className="field">
            <span className="field-label">
              Why is this wrong?
              <span className="muted">
                {" "}
                — this becomes a permanent reasoning rule for{" "}
                {item.agent}
              </span>
            </span>
            <textarea
              rows={3}
              value={reason}
              autoFocus
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g. This trial was terminated early due to COVID — terminated trials are exempt from result posting requirements."
            />
          </label>
          <div className="form-actions">
            <button
              className="btn reject"
              disabled={busy || reason.trim().length < 10}
              onClick={() => void commit("reject")}
            >
              {busy ? "Saving…" : "Reject and teach the agent"}
            </button>
            <button className="btn ghost" disabled={busy} onClick={() => setMode(null)}>
              Cancel
            </button>
          </div>
          {reason.trim().length > 0 && reason.trim().length < 10 && (
            <p className="note warn">
              Give a specific reason — a vague rule makes the agent worse, not
              better.
            </p>
          )}
        </div>
      )}

      {mode === "edit" && (
        <div className="decision-form">
          <label className="field">
            <span className="field-label">Corrected summary</span>
            <textarea
              rows={3}
              value={editSummary}
              autoFocus
              onChange={(e) => setEditSummary(e.target.value)}
            />
          </label>
          <div className="form-actions">
            <button
              className="btn primary"
              disabled={busy || !editSummary.trim()}
              onClick={() => void commit("edit")}
            >
              {busy ? "Saving…" : "Save and approve"}
            </button>
            <button className="btn ghost" disabled={busy} onClick={() => setMode(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {error && <ErrorBox error={error} />}
    </article>
  );
}
