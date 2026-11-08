import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { TicketDetailResponse, SimilarTicketsResponse } from '../types';
import { ProcessingBadge } from './ProcessingBadge';

interface TicketDetailViewProps {
  ticketId: string;
  onBack: () => void;
  onSelectTicket?: (ticketId: string) => void;
}

export const TicketDetailView = ({ ticketId, onBack, onSelectTicket }: TicketDetailViewProps) => {
  const [data, setData] = useState<TicketDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [similarData, setSimilarData] = useState<SimilarTicketsResponse | null>(null);
  const [similarLoading, setSimilarLoading] = useState(true);
  const [similarError, setSimilarError] = useState<string | null>(null);

  const currentTicketIdRef = useRef(ticketId);
  currentTicketIdRef.current = ticketId;

  const loadTicketData = async (targetTicketId: string, isManualRefresh = false) => {
    if (isManualRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
      setData(null);
      setSimilarData(null);
    }
    setError(null);
    setSimilarLoading(true);
    setSimilarError(null);

    try {
      const [ticketRes, similarRes] = await Promise.all([
        api.getTicketDetail(targetTicketId),
        api.getSimilarTickets(targetTicketId).catch((err) => {
          if (currentTicketIdRef.current === targetTicketId) {
            setSimilarError(err instanceof Error ? err.message : 'Failed to fetch similar tickets.');
          }
          return null;
        }),
      ]);

      if (currentTicketIdRef.current !== targetTicketId) {
        return;
      }

      setData(ticketRes);
      if (similarRes && similarRes.ticket_id === targetTicketId) {
        setSimilarData(similarRes);
      } else if (!similarRes) {
        setSimilarData(null);
      }
    } catch (err) {
      if (currentTicketIdRef.current === targetTicketId) {
        setError(err instanceof Error ? err.message : 'Failed to load ticket.');
      }
    } finally {
      if (currentTicketIdRef.current === targetTicketId) {
        setSimilarLoading(false);
        if (isManualRefresh) {
          setRefreshing(false);
        } else {
          setLoading(false);
        }
      }
    }
  };

  useEffect(() => {
    currentTicketIdRef.current = ticketId;
    loadTicketData(ticketId, false);
  }, [ticketId]);

  const handleRefresh = () => {
    loadTicketData(ticketId, true);
  };

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

  const { ticket, prediction, processing, notices } = data;

  // Sort decision margin scores descending if available
  const sortedScores = prediction?.decision_scores
    ? Object.entries(prediction.decision_scores).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div>
      {/* Top Action Bar: Back button and Refresh button */}
      <div
        style={{
          marginBottom: '20px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
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

        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleRefresh}
          disabled={refreshing || loading}
          style={{ padding: '6px 12px', fontSize: '13px', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
        >
          {refreshing ? 'Refreshing...' : '↻ Refresh'}
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
        {/* Left: Consumer Complaint Narrative & Similar Tickets */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
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

          {/* Similar Tickets Section */}
          <div className="card" style={{ padding: '24px' }}>
            <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-main)' }}>
                  Similar Tickets
                </h2>
                {similarData?.model_name && (
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    Selected embedding model: {similarData.model_name}
                  </span>
                )}
              </div>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
                Candidate matches based on exact cosine similarity of sentence embeddings.
              </p>
            </div>

            {similarLoading ? (
              <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                Finding possible similar tickets...
              </div>
            ) : similarError ? (
              <div className="alert alert-danger" style={{ margin: 0, fontSize: '13px' }}>
                Unable to load similar tickets: {similarError}
              </div>
            ) : !similarData?.has_embedding || similarData?.embedding_status === 'not_ready' ? (
              <div
                style={{
                  backgroundColor: 'var(--bg-subtle)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '4px',
                  padding: '16px',
                  textAlign: 'center',
                }}
              >
                <p style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-main)', marginBottom: '4px' }}>
                  Embedding Not Ready
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  {similarData?.message ||
                    'No embedding vector has been generated for this ticket yet. Similar ticket suggestions will become available once background embedding generation is enabled and completed.'}
                </p>
              </div>
            ) : similarData.items.length === 0 ? (
              <div
                style={{
                  backgroundColor: 'var(--bg-subtle)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '4px',
                  padding: '16px',
                  textAlign: 'center',
                }}
              >
                <p style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-main)', marginBottom: '4px' }}>
                  No Similar Tickets Found
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  No other tickets in this tenant share embeddings generated by the selected embedding model ({similarData.model_name}).
                </p>
              </div>
            ) : (
              <div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left' }}>
                        <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600 }}>Subject</th>
                        <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, width: '90px' }}>Status</th>
                        <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, textAlign: 'right', width: '140px' }}>
                          Cosine Similarity
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {similarData.items.map((item) => (
                        <tr key={item.ticket_id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                          <td style={{ padding: '10px' }}>
                            {onSelectTicket ? (
                              <button
                                type="button"
                                onClick={() => onSelectTicket(item.ticket_id)}
                                style={{
                                  background: 'none',
                                  border: 'none',
                                  padding: 0,
                                  color: 'var(--accent)',
                                  fontWeight: 500,
                                  fontSize: '13px',
                                  cursor: 'pointer',
                                  textAlign: 'left',
                                }}
                              >
                                {item.subject}
                              </button>
                            ) : (
                              <span style={{ fontWeight: 500, color: 'var(--text-main)' }}>{item.subject}</span>
                            )}
                            {item.external_id && (
                              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                                Ref: <code className="text-mono">{item.external_id}</code>
                              </div>
                            )}
                          </td>
                          <td style={{ padding: '10px' }}>
                            <span className={`badge badge-status-${item.status.toLowerCase()}`}>{item.status}</span>
                          </td>
                          <td style={{ padding: '10px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontWeight: 500 }}>
                            <span
                              title="Exact cosine similarity between embedding vectors [-1.0, 1.0]. Not a calibrated probability."
                              style={{
                                color: item.similarity_score >= 0.8 ? 'var(--primary)' : 'var(--text-main)',
                              }}
                            >
                              {item.similarity_score.toFixed(4)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '12px', lineHeight: 1.4 }}>
                  <strong>Notice</strong>: {similarData.notice}
                </p>
              </div>
            )}
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

          {/* Card: Background Processing Status */}
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
                Processing Status
              </h2>
              <ProcessingBadge status={processing} />
            </div>

            <div style={{ fontSize: '13px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  borderBottom: processing?.details ? '1px solid var(--border-color)' : 'none',
                  paddingBottom: processing?.details ? '8px' : '0',
                }}
              >
                <span className="text-muted">Queue Status:</span>
                <strong style={{ textTransform: 'capitalize' }}>
                  {processing?.state ? processing.state.replace('_', ' ') : 'Not Found'}
                </strong>
              </div>

              {processing?.details && (
                <div
                  style={{
                    marginTop: '8px',
                    fontSize: '12px',
                    color: processing.state === 'failed' ? 'var(--danger)' : 'var(--text-muted)',
                    lineHeight: 1.4,
                  }}
                >
                  {processing.details}
                </div>
              )}
            </div>
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
              ) : processing?.state === 'failed' ? (
                <span className="badge badge-failed">Unavailable</span>
              ) : processing?.state === 'completed' || processing?.state === 'not_found' ? (
                <span className="badge badge-not-found">None</span>
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
                <p style={{ fontSize: '13px', color: 'var(--text-main)', marginBottom: '8px', fontWeight: 500 }}>
                  {processing?.state === 'failed'
                    ? 'Automated prediction unavailable; review required.'
                    : processing?.state === 'completed'
                    ? 'No model prediction was generated for this ticket.'
                    : processing?.state === 'not_found'
                    ? 'No automated processing job was found for this ticket.'
                    : 'Model prediction is pending background processing.'}
                </p>
                <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  {processing?.state === 'failed'
                    ? 'Background processing encountered an error and could not complete automated triage.'
                    : processing?.state === 'completed'
                    ? 'Worker completed processing without producing a category prediction.'
                    : processing?.state === 'not_found'
                    ? 'This ticket was created without an associated ingestion job.'
                    : 'Inference occurs asynchronously via background worker jobs. Use the Refresh button above to check for updates.'}
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
