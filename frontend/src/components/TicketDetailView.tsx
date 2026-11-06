import { useEffect, useState } from 'react';
import { api } from '../api';
import { TicketDetailResponse } from '../types';

interface TicketDetailViewProps {
  ticketId: string;
  onBack: () => void;
}

export const TicketDetailView = ({ ticketId, onBack }: TicketDetailViewProps) => {
  const [data, setData] = useState<TicketDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    api
      .getTicketDetail(ticketId)
      .then((res) => {
        if (isMounted) setData(res);
      })
      .catch((err) => {
        if (isMounted) setError(err instanceof Error ? err.message : 'Failed to load ticket.');
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [ticketId]);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '48px 0', color: 'var(--text-muted)' }}>
        Loading ticket details...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div>
        <div style={{ marginBottom: '20px' }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onBack}
            style={{ padding: '6px 12px', fontSize: '13px' }}
          >
            ← Back to Ticket Queue
          </button>
        </div>
        <div className="alert alert-danger" role="alert">
          {error || 'Ticket not found.'}
        </div>
      </div>
    );
  }

  const { ticket, prediction, notices } = data;

  // Sort decision margin scores descending if available
  const sortedScores = prediction?.decision_scores
    ? Object.entries(prediction.decision_scores).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div>
      {/* Back button */}
      <div style={{ marginBottom: '20px' }}>
        <button
          type="button"
          onClick={onBack}
          style={{
            background: 'none',
            border: 'none',
            padding: 0,
            textDecoration: 'none',
            color: 'var(--accent)',
            fontSize: '13px',
            fontWeight: 500,
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
          }}
        >
          ← Back to Ticket Queue
        </button>
      </div>

      {/* Header */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
          <h1 style={{ fontSize: '22px', fontWeight: 600, color: 'var(--text-main)' }}>{ticket.subject}</h1>
          <span className={`badge badge-status-${ticket.status.toLowerCase()}`}>{ticket.status}</span>
        </div>
        <div style={{ fontSize: '13px', color: 'var(--text-muted)', display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
          <span>
            ID: <code className="text-mono">{ticket.id}</code>
          </span>
          {ticket.external_id && (
            <span>
              External Ref: <code className="text-mono">{ticket.external_id}</code>
            </span>
          )}
          {ticket.customer_email && (
            <span>
              Customer: <strong style={{ color: 'var(--text-main)' }}>{ticket.customer_email}</strong>
            </span>
          )}
          <span>
            Received:{' '}
            {ticket.created_at ? new Date(ticket.created_at).toISOString().replace('T', ' ').slice(0, 19) + ' UTC' : '—'}
          </span>
        </div>
      </div>

      {/* Two-Column Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr) 380px',
          gap: '24px',
          alignItems: 'start',
        }}
      >
        {/* Left: Consumer Complaint Narrative */}
        <div className="card" style={{ padding: '24px' }}>
          <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '16px' }}>
            <h2 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-main)' }}>
              Consumer Complaint Narrative
            </h2>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Full customer submission as received by the ingestion pipeline.
            </p>
          </div>

          {/* Render customer content strictly as text (no dangerouslySetInnerHTML) */}
          <div
            style={{
              backgroundColor: 'var(--bg-subtle)',
              border: '1px solid var(--border-color)',
              borderRadius: '4px',
              padding: '16px',
              fontSize: '13px',
              lineHeight: 1.6,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              color: 'var(--text-main)',
              fontFamily: 'var(--font-sans)',
            }}
          >
            {ticket.body}
          </div>
        </div>

        {/* Right: Triage & Model Diagnostics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Card 1: Human Triage Assignment */}
          <div className="card" style={{ padding: '20px' }}>
            <h2
              style={{
                fontSize: '14px',
                fontWeight: 600,
                color: 'var(--text-main)',
                marginBottom: '12px',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
            >
              Human Triage Assignment
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  borderBottom: '1px solid var(--border-color)',
                  paddingBottom: '8px',
                }}
              >
                <span className="text-muted">Assigned Category:</span>
                <strong>{ticket.category || 'None (Unassigned)'}</strong>
              </div>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  borderBottom: '1px solid var(--border-color)',
                  paddingBottom: '8px',
                }}
              >
                <span className="text-muted">Assigned Priority:</span>
                <strong>{ticket.priority || 'None (Unassigned)'}</strong>
              </div>
            </div>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '10px', lineHeight: 1.4 }}>
              Human-assigned labels represent final agent triage and are tracked independently from automated model
              predictions.
            </p>
          </div>

          {/* Card 2: Automated Model Prediction */}
          <div className="card" style={{ padding: '20px' }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '12px',
              }}
            >
              <h2
                style={{
                  fontSize: '14px',
                  fontWeight: 600,
                  color: 'var(--text-main)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}
              >
                Automated Model Prediction
              </h2>
              {prediction ? (
                <span className="badge badge-pred">Active</span>
              ) : (
                <span className="badge badge-pending">Prediction Pending</span>
              )}
            </div>

            {prediction ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '13px' }}>
                <div>
                  <div className="text-muted" style={{ fontSize: '12px', marginBottom: '4px' }}>
                    Predicted Product:
                  </div>
                  <div style={{ fontWeight: 600, fontSize: '14px', color: 'var(--primary)' }}>
                    {prediction.predicted_label}
                  </div>
                </div>

                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    borderBottom: '1px solid var(--border-color)',
                    paddingBottom: '8px',
                  }}
                >
                  <span className="text-muted">Model Version:</span>
                  <span className="text-mono" style={{ fontSize: '12px' }}>
                    {prediction.model_name} ({prediction.model_version})
                  </span>
                </div>

                <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span className="text-muted">Confidence:</span>
                    <span style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
                      {prediction.confidence_display}
                    </span>
                  </div>
                  <p style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.3 }}>
                    {notices.confidence_notice}
                  </p>
                </div>

                {sortedScores.length > 0 && (
                  <div style={{ marginTop: '4px' }}>
                    <div style={{ fontSize: '12px', marginBottom: '8px', fontWeight: 500, color: 'var(--text-muted)' }}>
                      Decision Margin Scores:
                    </div>
                    <div
                      style={{
                        maxHeight: '220px',
                        overflowY: 'auto',
                        border: '1px solid var(--border-color)',
                        borderRadius: '4px',
                      }}
                    >
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
                        <tbody>
                          {sortedScores.map(([label, score]) => (
                            <tr
                              key={label}
                              style={{
                                borderBottom: '1px solid var(--border-color)',
                                backgroundColor:
                                  label === prediction.predicted_label ? '#f5f3ff' : 'transparent',
                                fontWeight: label === prediction.predicted_label ? 600 : 'normal',
                              }}
                            >
                              <td style={{ padding: '6px 8px', color: 'var(--text-main)' }}>{label}</td>
                              <td
                                style={{
                                  padding: '6px 8px',
                                  textAlign: 'right',
                                  fontFamily: 'var(--font-mono)',
                                  whiteSpace: 'nowrap',
                                  color: 'var(--text-muted)',
                                }}
                              >
                                {score >= 0 ? `+${score.toFixed(4)}` : score.toFixed(4)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ padding: '16px 0', textAlign: 'center' }}>
                <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                  No model prediction record exists for this ticket.
                </p>
                <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Inference occurs asynchronously via background worker jobs. Check worker process status if jobs remain
                  pending.
                </p>
              </div>
            )}
          </div>

          {/* Card 3: Priority Model Notice */}
          <div className="card" style={{ padding: '16px 20px', backgroundColor: 'var(--bg-subtle)' }}>
            <div
              style={{
                fontSize: '12px',
                fontWeight: 600,
                color: 'var(--text-muted)',
                marginBottom: '4px',
                textTransform: 'uppercase',
              }}
            >
              Priority Status
            </div>
            <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-main)', marginBottom: '4px' }}>
              {notices.priority_notice}
            </div>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
              Automated priority prediction remains intentionally deferred. No heuristic or placeholder priority is
              assigned.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

