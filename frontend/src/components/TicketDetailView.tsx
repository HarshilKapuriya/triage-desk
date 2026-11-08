import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { TicketDetailResponse, SimilarTicketsResponse, RelevantKnowledgeResponse } from '../types';
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

  const [knowledgeData, setKnowledgeData] = useState<RelevantKnowledgeResponse | null>(null);
  const [knowledgeLoading, setKnowledgeLoading] = useState(true);
  const [knowledgeError, setKnowledgeError] = useState<string | null>(null);

  const [rowPendingId, setRowPendingId] = useState<string | null>(null);
  const [rowErrors, setRowErrors] = useState<{ [ticketId: string]: string }>({});

  const currentTicketIdRef = useRef(ticketId);
  currentTicketIdRef.current = ticketId;

  const loadTicketData = async (targetTicketId: string, isManualRefresh = false) => {
    if (isManualRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
      setData(null);
      setSimilarData(null);
      setKnowledgeData(null);
    }
    setError(null);
    setSimilarLoading(true);
    setSimilarError(null);
    setKnowledgeLoading(true);
    setKnowledgeError(null);
    setRowPendingId(null);
    setRowErrors({});

    try {
      const [ticketRes, similarRes, knowledgeRes] = await Promise.all([
        api.getTicketDetail(targetTicketId),
        api.getSimilarTickets(targetTicketId).catch((err) => {
          if (currentTicketIdRef.current === targetTicketId) {
            setSimilarError(err instanceof Error ? err.message : 'Failed to fetch similar tickets.');
          }
          return null;
        }),
        api.getRelevantKnowledge(targetTicketId).catch((err) => {
          if (currentTicketIdRef.current === targetTicketId) {
            setKnowledgeError(err instanceof Error ? err.message : 'Failed to fetch relevant knowledge.');
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

      if (knowledgeRes && knowledgeRes.ticket_id === targetTicketId) {
        setKnowledgeData(knowledgeRes);
      } else if (!knowledgeRes) {
        setKnowledgeData(null);
      }
    } catch (err) {
      if (currentTicketIdRef.current === targetTicketId) {
        setError(err instanceof Error ? err.message : 'Failed to load ticket.');
      }
    } finally {
      if (currentTicketIdRef.current === targetTicketId) {
        setSimilarLoading(false);
        setKnowledgeLoading(false);
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

  const handleFeedback = async (similarTicketId: string, decision: 'confirmed' | 'rejected') => {
    if (rowPendingId) return;
    const originTicketId = ticketId;
    setRowPendingId(similarTicketId);
    setRowErrors((prev) => {
      const next = { ...prev };
      delete next[similarTicketId];
      return next;
    });

    try {
      const res = await api.submitSimilarFeedback(ticketId, similarTicketId, decision);
      if (currentTicketIdRef.current !== originTicketId) {
        return; // Discard response if user navigated to a different ticket
      }
      setSimilarData((prev) => {
        if (!prev || currentTicketIdRef.current !== originTicketId || prev.ticket_id !== originTicketId) return prev;
        return {
          ...prev,
          items: prev.items.map((it) => {
            if (it.ticket_id === similarTicketId) {
              return {
                ...it,
                decision_status: res.decision,
                reviewed_at: res.reviewed_at,
              };
            }
            return it;
          }),
        };
      });
    } catch (err: unknown) {
      if (currentTicketIdRef.current === originTicketId) {
        const msg = err instanceof Error ? err.message : 'Failed to record decision.';
        setRowErrors((prev) => ({ ...prev, [similarTicketId]: msg }));
      }
    } finally {
      if (currentTicketIdRef.current === originTicketId) {
        setRowPendingId(null);
      }
    }
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
                        <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, width: '80px' }}>Status</th>
                        <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, textAlign: 'right', width: '110px' }}>
                          Cosine Similarity
                        </th>
                        <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, textAlign: 'right', width: '200px' }}>
                          Human Decision
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
                          <td style={{ padding: '10px', textAlign: 'right', verticalAlign: 'middle' }}>
                            {item.decision_status === 'confirmed' ? (
                              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', justifyContent: 'flex-end' }}>
                                <span
                                  style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '4px',
                                    padding: '2px 8px',
                                    borderRadius: '4px',
                                    fontSize: '11px',
                                    fontWeight: 600,
                                    backgroundColor: '#e6f4ea',
                                    color: '#137333',
                                    border: '1px solid #ceead6',
                                  }}
                                  title={item.reviewed_at ? `Confirmed on ${new Date(item.reviewed_at).toISOString().replace('T', ' ').slice(0, 19)} UTC` : 'Confirmed duplicate'}
                                >
                                  ✓ Confirmed
                                </span>
                                <button
                                  type="button"
                                  disabled={rowPendingId === item.ticket_id}
                                  onClick={() => handleFeedback(item.ticket_id, 'rejected')}
                                  style={{
                                    background: 'none',
                                    border: 'none',
                                    padding: '2px 4px',
                                    color: 'var(--text-muted)',
                                    fontSize: '11px',
                                    textDecoration: 'underline',
                                    cursor: rowPendingId === item.ticket_id ? 'not-allowed' : 'pointer',
                                  }}
                                  title="Change duplicate decision to Rejected"
                                >
                                  Reject
                                </button>
                              </div>
                            ) : item.decision_status === 'rejected' ? (
                              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', justifyContent: 'flex-end' }}>
                                <span
                                  style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '4px',
                                    padding: '2px 8px',
                                    borderRadius: '4px',
                                    fontSize: '11px',
                                    fontWeight: 500,
                                    backgroundColor: 'var(--bg-subtle)',
                                    color: 'var(--text-muted)',
                                    border: '1px solid var(--border-color)',
                                  }}
                                  title={item.reviewed_at ? `Rejected on ${new Date(item.reviewed_at).toISOString().replace('T', ' ').slice(0, 19)} UTC` : 'Rejected match'}
                                >
                                  ✗ Rejected
                                </span>
                                <button
                                  type="button"
                                  disabled={rowPendingId === item.ticket_id}
                                  onClick={() => handleFeedback(item.ticket_id, 'confirmed')}
                                  style={{
                                    background: 'none',
                                    border: 'none',
                                    padding: '2px 4px',
                                    color: 'var(--text-muted)',
                                    fontSize: '11px',
                                    textDecoration: 'underline',
                                    cursor: rowPendingId === item.ticket_id ? 'not-allowed' : 'pointer',
                                  }}
                                  title="Change duplicate decision to Confirmed"
                                >
                                  Confirm
                                </button>
                              </div>
                            ) : (
                              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', justifyContent: 'flex-end' }}>
                                <button
                                  type="button"
                                  className="btn btn-primary"
                                  disabled={rowPendingId === item.ticket_id}
                                  onClick={() => handleFeedback(item.ticket_id, 'confirmed')}
                                  style={{
                                    padding: '3px 8px',
                                    fontSize: '11px',
                                    fontWeight: 500,
                                    opacity: rowPendingId === item.ticket_id ? 0.6 : 1,
                                  }}
                                >
                                  {rowPendingId === item.ticket_id ? 'Saving...' : 'Confirm'}
                                </button>
                                <button
                                  type="button"
                                  className="btn btn-secondary"
                                  disabled={rowPendingId === item.ticket_id}
                                  onClick={() => handleFeedback(item.ticket_id, 'rejected')}
                                  style={{
                                    padding: '3px 8px',
                                    fontSize: '11px',
                                    fontWeight: 500,
                                    opacity: rowPendingId === item.ticket_id ? 0.6 : 1,
                                  }}
                                >
                                  Reject
                                </button>
                              </div>
                            )}
                            {rowErrors[item.ticket_id] && (
                              <div style={{ color: 'var(--danger)', fontSize: '11px', marginTop: '4px', textAlign: 'right' }}>
                                {rowErrors[item.ticket_id]}
                              </div>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '12px', lineHeight: 1.4 }}>
                  <strong>Notice</strong>: {similarData.notice} Human confirmation records a duplicate decision only; tickets are not merged or closed, and feedback does not alter model predictions automatically.
                </p>
              </div>
            )}
          </div>

          {/* Relevant Knowledge Section */}
          <div className="card" style={{ padding: '24px' }}>
            <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-main)', margin: 0 }}>
                  Relevant Knowledge
                </h2>
                {knowledgeData?.model_name && (
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    Embedding model: {knowledgeData.model_name}
                  </span>
                )}
              </div>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px', marginBottom: 0 }}>
                Candidate knowledge-base excerpts ranked by cosine similarity against this complaint.
              </p>
            </div>

            {knowledgeLoading ? (
              <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                Retrieving relevant knowledge excerpts...
              </div>
            ) : knowledgeError ? (
              <div className="alert alert-danger" style={{ margin: 0, fontSize: '13px' }}>
                Unable to load relevant knowledge: {knowledgeError}
              </div>
            ) : knowledgeData?.status === 'missing_ticket_embedding' ? (
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
                  {knowledgeData.message ||
                    'No embedding vector has been generated for this ticket yet. Knowledge retrieval requires an embedded ticket.'}
                </p>
              </div>
            ) : knowledgeData?.status === 'no_chunks_available' ? (
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
                  No Knowledge Base Content
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  {knowledgeData.message ||
                    'No knowledge-base articles or passages exist for this tenant. Visit Knowledge Base to add articles.'}
                </p>
              </div>
            ) : knowledgeData?.status === 'articles_processing' ? (
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
                  Articles Still Processing
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  {knowledgeData.message ||
                    'Knowledge-base articles have been ingested, but passage chunk embeddings are still processing or pending background queue activation.'}
                </p>
              </div>
            ) : knowledgeData?.status === 'articles_failed' ? (
              <div
                style={{
                  backgroundColor: '#fef2f2',
                  border: '1px solid #fecaca',
                  borderRadius: '4px',
                  padding: '16px',
                  textAlign: 'center',
                }}
              >
                <p style={{ fontSize: '13px', fontWeight: 500, color: '#b91c1c', marginBottom: '4px' }}>
                  Knowledge Embedding Failed
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  {knowledgeData.message ||
                    'Tenant knowledge-base articles failed background embedding. Please check the Knowledge Base view.'}
                </p>
              </div>
            ) : knowledgeData?.status === 'model_mismatch' ? (
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
                  Embedding Model Incompatible
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  {knowledgeData.message ||
                    'Existing knowledge-base chunks were generated using a different model or version than this ticket.'}
                </p>
              </div>
            ) : !knowledgeData?.items || knowledgeData.items.length === 0 ? (
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
                  No Relevant Excerpts Found
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  No knowledge-base passages qualified for this complaint under the active embedding model.
                </p>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {knowledgeData.items.map((item) => (
                    <div
                      key={item.chunk_id}
                      style={{
                        backgroundColor: 'var(--bg-subtle)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '6px',
                        padding: '14px',
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'baseline',
                          marginBottom: '6px',
                        }}
                      >
                        <div>
                          <strong style={{ fontSize: '13px', color: 'var(--text-main)' }}>
                            {item.article_title}
                          </strong>
                          <span
                            style={{
                              marginLeft: '8px',
                              fontSize: '11px',
                              color: 'var(--text-muted)',
                              fontFamily: 'var(--font-mono)',
                            }}
                          >
                            Article: <code className="text-mono">{item.article_id.slice(0, 8)}</code> · Chunk #{item.chunk_index + 1} (<code className="text-mono">{item.chunk_id.slice(0, 8)}</code>)
                          </span>
                        </div>
                        <span
                          title="Exact cosine similarity between ticket embedding and knowledge-base chunks [-1.0, 1.0]. Not a calibrated confidence score."
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '12px',
                            fontWeight: 600,
                            color: item.similarity_score >= 0.75 ? 'var(--primary)' : 'var(--text-main)',
                          }}
                        >
                          Cosine Sim: {item.similarity_score.toFixed(4)}
                        </span>
                      </div>
                      <div
                        style={{
                          fontSize: '12px',
                          lineHeight: 1.5,
                          color: 'var(--text-main)',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          backgroundColor: 'var(--bg-card)',
                          padding: '10px',
                          borderRadius: '4px',
                          border: '1px solid var(--border-color)',
                        }}
                      >
                        {item.excerpt}
                      </div>
                      <div
                        style={{
                          fontSize: '10px',
                          color: 'var(--text-muted)',
                          marginTop: '6px',
                          display: 'flex',
                          gap: '12px',
                        }}
                      >
                        <span>
                          Article Ref: <code className="text-mono">{item.article_id.slice(0, 8)}</code>
                        </span>
                        <span>Tokens: {item.token_count}</span>
                        <span>Chars: {item.char_count}</span>
                      </div>
                    </div>
                  ))}
                </div>

                <p
                  style={{
                    fontSize: '11px',
                    color: 'var(--text-muted)',
                    marginTop: '14px',
                    marginBottom: 0,
                    lineHeight: 1.4,
                  }}
                >
                  <strong>Notice</strong>: {knowledgeData.notice}
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
