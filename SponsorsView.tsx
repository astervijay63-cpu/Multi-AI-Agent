import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { SponsorProfile } from "../types";
import { Empty, ErrorBox, Spinner, formatDate } from "./ui";

export function SponsorsView() {
  const [sponsors, setSponsors] = useState<SponsorProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(50);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.sponsors(limit);
      setSponsors(response.sponsors);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="view">
      <header className="view-header">
        <div>
          <h2>Sponsor credibility</h2>
          <p className="sub">
            Reputation profiles built up across runs, least credible first.
          </p>
        </div>
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
      </header>

      {error && <ErrorBox error={error} onRetry={() => void load()} />}
      {loading && <Spinner label="Loading sponsors…" />}
      {!loading && !error && sponsors.length === 0 && (
        <Empty>
          No sponsor profiles yet. They are created as agents analyse studies.
        </Empty>
      )}

      {sponsors.length > 0 && (
        <div className="table-wrap panel">
          <table>
            <thead>
              <tr>
                <th className="wide">Sponsor</th>
                <th>Credibility</th>
                <th>Studies</th>
                <th>Results posted</th>
                <th>Results missing</th>
                <th>Broken promises</th>
                <th>Avg delay</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {sponsors.map((s) => (
                <tr key={s.sponsor}>
                  <td className="wide">{s.sponsor}</td>
                  <td>
                    <CredibilityBadge score={s.credibility_score} />
                  </td>
                  <td>{s.total_studies}</td>
                  <td>{s.results_posted}</td>
                  <td className={s.results_missing > 0 ? "bad" : ""}>
                    {s.results_missing}
                  </td>
                  <td className={s.broken_promises > 0 ? "bad" : ""}>
                    {s.broken_promises}
                  </td>
                  <td>
                    {s.avg_delay_days
                      ? `${Math.round(s.avg_delay_days)} days`
                      : "—"}
                  </td>
                  <td className="muted nowrap">{formatDate(s.last_updated)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// The track record agent treats anything under 0.6 as a concerning
// sponsor and under 0.4 as a strong signal, so the bands match those cuts.
function CredibilityBadge({ score }: { score: number }) {
  const band = score >= 0.6 ? "high" : score >= 0.4 ? "mid" : "low";
  return <span className={`cred cred-${band}`}>{score.toFixed(2)}</span>;
}
