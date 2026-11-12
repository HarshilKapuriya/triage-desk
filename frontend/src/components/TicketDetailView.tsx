import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import {
  TicketDetailResponse,
  SimilarTicketsResponse,
  RelevantKnowledgeResponse,
  TicketSLAResponse,
  TicketEventsResponse,
} from '../types';
import { ProcessingBadge } from './ProcessingBadge';

interface TicketDetailViewProps {
  ticketId: string;
  onBack: () => void;
  onSelectTicket?: (ticketId: string) => void;
  slaPolicyVersion?: number;
}

const formatDuration = (mins: number) => {
  if (mins < 60) return `${mins}m`;
  const hours = (mins / 60).toFixed(1).replace('.0', '');
  if (mins < 1440) return `${mins}m (${hours}h)`;
  const days = (mins / 1440).toFixed(1).replace('.0', '');
  return `${mins}m (${days}d)`;
};

const formatUtcTimestamp = (isoStr: string | null | undefined): string => {
  if (!isoStr) return '—';
  const parsed = new Date(isoStr);
  if (Number.isNaN(parsed.getTime())) {
    return isoStr;
  }
  return `${parsed.toISOString().replace('T', ' ').slice(0, 19)} UTC`;
};

const formatElapsedSeconds = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  return formatDuration(Math.round(seconds / 60));
};

export const TicketDetailView = ({
  ticketId,
  onBack,
  onSelectTicket,
  slaPolicyVersion,
}: TicketDetailViewProps) => {
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

  const [slaData, setSlaData] = useState<TicketSLAResponse | null>(null);
  const [slaLoading, setSlaLoading] = useState(true);
  const [slaError, setSlaError] = useState<string | null>(null);

  const [eventsData, setEventsData] = useState<TicketEventsResponse | null>(null);
  const [eventsLoading, setEventsLoading] = useState(true);
  const [eventsError, setEventsError] = useState<string | null>(null);

  const [lifecyclePending, setLifecyclePending] = useState(false);
  const [lifecycleError, setLifecycleError] = useState<string | null>(null);

  const [rowPendingId, setRowPendingId] = useState<string | null>(null);
  const [rowErrors, setRowErrors] = useState<{ [ticketId: string]: string }>({});

  const currentTicketIdRef = useRef(ticketId);
  currentTicketIdRef.current = ticketId;
  const requestSeqRef = useRef(0);
  const eventsSeqRef = useRef(0);
  const eventsAbortRef = useRef<AbortController | null>(null);

  const loadTicketData = async (targetTicketId: string, isManualRefresh = false) => {
    const seq = ++requestSeqRef.current;
    const evSeq = ++eventsSeqRef.current;
    eventsAbortRef.current?.abort();
    const eventsController = new AbortController();
    eventsAbortRef.current = eventsController;

    if (isManualRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
      setData(null);
      setSimilarData(null);
      setKnowledgeData(null);
      setSlaData(null);
      setEventsData(null);
    }
    setError(null);
    setSimilarLoading(true);
    setSimilarError(null);
    setKnowledgeLoading(true);
    setKnowledgeError(null);
    setSlaLoading(true);
    setSlaError(null);
    setEventsLoading(true);
    setEventsError(null);
    setLifecycleError(null);
    setRowPendingId(null);
    setRowErrors({});

    try {
      const [ticketRes, similarRes, knowledgeRes, slaRes, eventsRes] = await Promise.all([
        api.getTicketDetail(targetTicketId),
        api.getSimilarTickets(targetTicketId).catch((err) => {
          if (requestSeqRef.current === seq && currentTicketIdRef.current === targetTicketId) {
            setSimilarError(err instanceof Error ? err.message : 'Failed to fetch similar tickets.');
          }
          return null;
        }),
        api.getRelevantKnowledge(targetTicketId).catch((err) => {
          if (requestSeqRef.current === seq && currentTicketIdRef.current === targetTicketId) {
            setKnowledgeError(err instanceof Error ? err.message : 'Failed to fetch relevant knowledge.');
          }
          return null;
        }),
        api.getTicketSla(targetTicketId).catch((err) => {
          if (requestSeqRef.current === seq && currentTicketIdRef.current === targetTicketId) {
            setSlaError(err instanceof Error ? err.message : 'Failed to fetch SLA evaluation.');
          }
          return null;
        }),
        api.getTicketEvents(targetTicketId, eventsController.signal).catch((err) => {
          if (err instanceof DOMException && err.name === 'AbortError') {
            return null;
          }
          if (eventsSeqRef.current === evSeq && currentTicketIdRef.current === targetTicketId) {
            setEventsError(err instanceof Error ? err.message : 'Failed to load lifecycle timeline.');
          }
          return null;
        }),
      ]);

      if (requestSeqRef.current !== seq || currentTicketIdRef.current !== targetTicketId) {
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

      if (slaRes) {
        setSlaData(slaRes);
      }

      if (
        eventsSeqRef.current === evSeq &&
        eventsRes &&
        eventsRes.ticket_id === targetTicketId
      ) {
        setEventsData(eventsRes);
      }
    } catch (err) {
      if (requestSeqRef.current === seq && currentTicketIdRef.current === targetTicketId) {
        setError(err instanceof Error ? err.message : 'Failed to load ticket.');
      }
    } finally {
      if (eventsSeqRef.current === evSeq && currentTicketIdRef.current === targetTicketId) {
        setEventsLoading(false);
      }
      if (requestSeqRef.current === seq && currentTicketIdRef.current === targetTicketId) {
        setSimilarLoading(false);
        setKnowledgeLoading(false);
        setSlaLoading(false);
        if (isManualRefresh) {
          setRefreshing(false);
        } else {
          setLoading(false);
        }
      }
    }
  };

  const refreshEventsOnly = async (targetTicketId: string) => {
    const evSeq = ++eventsSeqRef.current;
    eventsAbortRef.current?.abort();
    const controller = new AbortController();
    eventsAbortRef.current = controller;

    setEventsLoading(true);
    setEventsError(null);
    try {
      const res = await api.getTicketEvents(targetTicketId, controller.signal);
      if (
        eventsSeqRef.current === evSeq &&
        currentTicketIdRef.current === targetTicketId &&
        res.ticket_id === targetTicketId
      ) {
        setEventsData(res);
        setEventsError(null);
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return;
      }
      if (eventsSeqRef.current === evSeq && currentTicketIdRef.current === targetTicketId) {
        setEventsError(err instanceof Error ? err.message : 'Failed to load lifecycle timeline.');
      }
    } finally {
      if (eventsSeqRef.current === evSeq && currentTicketIdRef.current === targetTicketId) {
        setEventsLoading(false);
      }
    }
  };

  const refreshSlaOnly = async (targetTicketId: string) => {
    const seq = ++requestSeqRef.current;
    try {
      const res = await api.getTicketSla(targetTicketId);
      if (requestSeqRef.current === seq && currentTicketIdRef.current === targetTicketId) {
        setSlaData(res);
        setSlaError(null);
      }
    } catch (err: unknown) {
      if (requestSeqRef.current === seq && currentTicketIdRef.current === targetTicketId) {
        // Discard silent SLA refresh errors to avoid flickering
      }
    }
  };

  const handleRecordFirstResponse = async () => {
    const originId = ticketId;
    setLifecyclePending(true);
    setLifecycleError(null);
    try {
      const res = await api.recordFirstResponse(originId);
      if (currentTicketIdRef.current === originId) {
        setData((prev) => {
          if (!prev) return null;
          return {
            ...prev,
            ticket: {
              ...prev.ticket,
              first_response_at: res.ticket.first_response_at,
            },
          };
        });
        await Promise.all([refreshSlaOnly(originId), refreshEventsOnly(originId)]);
      }
    } catch (err: unknown) {
      if (currentTicketIdRef.current === originId) {
        setLifecycleError(err instanceof Error ? err.message : 'Failed to record first response.');
      }
    } finally {
      if (currentTicketIdRef.current === originId) {
        setLifecyclePending(false);
      }
    }
  };

  const handleResolveTicket = async () => {
    const originId = ticketId;
    setLifecyclePending(true);
    setLifecycleError(null);
    try {
      const res = await api.resolveTicket(originId);
      if (currentTicketIdRef.current === originId) {
        setData((prev) => {
          if (!prev) return null;
          return {
            ...prev,
            ticket: {
              ...prev.ticket,
              status: res.ticket.status,
              resolved_at: res.ticket.resolved_at,
            },
          };
        });
        await Promise.all([refreshSlaOnly(originId), refreshEventsOnly(originId)]);
      }
    } catch (err: unknown) {
      if (currentTicketIdRef.current === originId) {
        setLifecycleError(err instanceof Error ? err.message : 'Failed to resolve ticket.');
      }
    } finally {
      if (currentTicketIdRef.current === originId) {
        setLifecyclePending(false);
      }
    }
  };

  const handleCloseTicket = async () => {
    const originId = ticketId;
    setLifecyclePending(true);
    setLifecycleError(null);
    try {
      const res = await api.closeTicket(originId);
      if (currentTicketIdRef.current === originId) {
        setData((prev) => {
          if (!prev) return null;
          return {
            ...prev,
            ticket: {
              ...prev.ticket,
              status: res.ticket.status,
            },
          };
        });
        await Promise.all([refreshSlaOnly(originId), refreshEventsOnly(originId)]);
      }
    } catch (err: unknown) {
      if (currentTicketIdRef.current === originId) {
        setLifecycleError(err instanceof Error ? err.message : 'Failed to close ticket.');
      }
    } finally {
      if (currentTicketIdRef.current === originId) {
        setLifecyclePending(false);
      }
    }
  };

  useEffect(() => {
    currentTicketIdRef.current = ticketId;
    loadTicketData(ticketId, false);
    return () => {
      requestSeqRef.current++;
      eventsSeqRef.current++;
      eventsAbortRef.current?.abort();
    };
  }, [ticketId]);

  useEffect(() => {
    if (slaPolicyVersion && slaPolicyVersion > 0) {
      refreshSlaOnly(ticketId);
    }
  }, [slaPolicyVersion, ticketId]);

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
          {ticket.first_response_at && (
            <span>
              First Response:{' '}
              <strong style={{ color: 'var(--text-main)' }}>
                {new Date(ticket.first_response_at).toISOString().replace('T', ' ').slice(0, 19)} UTC
              </strong>
            </span>
          )}
          {ticket.resolved_at && (
            <span>
              Resolved:{' '}
              <strong style={{ color: 'var(--text-main)' }}>
                {new Date(ticket.resolved_at).toISOString().replace('T', ' ').slice(0, 19)} UTC
              </strong>
            </span>
          )}
        </div>
      </div>

      {/* Lifecycle Actions Bar */}
      <div
        className="card"
        style={{
          marginBottom: '20px',
          padding: '14px 20px',
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '12px',
          backgroundColor: 'var(--bg-card)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span
            style={{
              fontSize: '12px',
              fontWeight: 600,
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            Agent Lifecycle Actions
          </span>
          {lifecyclePending && (
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Updating state...</span>
          )}
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
          {/* 1. Record First Response */}
          <button
            type="button"
            className="btn btn-secondary"
            disabled={
              lifecyclePending ||
              Boolean(ticket.first_response_at) ||
              ticket.status === 'resolved' ||
              ticket.status === 'closed'
            }
            onClick={handleRecordFirstResponse}
            style={{ padding: '6px 12px', fontSize: '13px' }}
            title={
              ticket.first_response_at
                ? `First response already recorded at ${new Date(ticket.first_response_at).toISOString().replace('T', ' ').slice(0, 19)} UTC`
                : ticket.status === 'resolved' || ticket.status === 'closed'
                ? `Cannot record first response on ${ticket.status} ticket`
                : 'Record customer response timestamp'
            }
          >
            {ticket.first_response_at ? '✓ First Response Recorded' : 'Record First Response'}
          </button>

          {/* 2. Resolve Ticket */}
          <button
            type="button"
            className="btn btn-secondary"
            disabled={
              lifecyclePending ||
              ticket.status === 'resolved' ||
              ticket.status === 'closed'
            }
            onClick={handleResolveTicket}
            style={{ padding: '6px 12px', fontSize: '13px' }}
            title={
              ticket.status === 'resolved'
                ? 'Ticket is already resolved'
                : ticket.status === 'closed'
                ? 'Closed ticket cannot be resolved'
                : 'Mark ticket as resolved'
            }
          >
            {ticket.status === 'resolved' ? '✓ Resolved' : 'Resolve Ticket'}
          </button>

          {/* 3. Close Ticket */}
          <button
            type="button"
            className="btn btn-secondary"
            disabled={lifecyclePending || ticket.status !== 'resolved'}
            onClick={handleCloseTicket}
            style={{ padding: '6px 12px', fontSize: '13px' }}
            title={
              ticket.status === 'closed'
                ? 'Ticket is already closed'
                : ticket.status !== 'resolved'
                ? `Ticket must be resolved before closing (current: ${ticket.status})`
                : 'Close resolved ticket'
            }
          >
            {ticket.status === 'closed' ? '✓ Closed' : 'Close Ticket'}
          </button>
        </div>
      </div>

      {lifecycleError && (
        <div className="alert alert-danger" style={{ marginBottom: '20px', fontSize: '13px' }}>
          {lifecycleError}
        </div>
      )}

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

          {/* Lifecycle Timeline Card */}
          <div className="card" style={{ padding: '24px' }}>
            <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-main)', margin: 0 }}>
                  Lifecycle Timeline
                </h2>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Factual lifecycle history
                </span>
              </div>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px', marginBottom: 0 }}>
                {eventsData?.ordering_notice ||
                  'Ordered by occurred_at ASC, recorded_at ASC, id ASC (stable event-time ordering, not guaranteed causal or commit visibility ordering).'}
              </p>
            </div>

            {eventsLoading ? (
              <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                Loading lifecycle timeline...
              </div>
            ) : eventsError ? (
              <div
                className="alert alert-danger"
                role="alert"
                style={{
                  margin: 0,
                  fontSize: '13px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '12px',
                  flexWrap: 'wrap',
                }}
              >
                <span>Unable to load lifecycle timeline: {eventsError}</span>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => refreshEventsOnly(ticket.id)}
                  disabled={eventsLoading}
                  style={{ padding: '4px 10px', fontSize: '12px' }}
                >
                  Retry Timeline
                </button>
              </div>
            ) : !eventsData?.schema_available ? (
              <div
                style={{
                  backgroundColor: 'var(--bg-subtle)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '4px',
                  padding: '16px',
                  textAlign: 'center',
                }}
              >
                <p style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
                  Lifecycle Timeline Unavailable
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  {eventsData?.message ||
                    'Migration 009 (ticket_events) is not provisioned in the database. Lifecycle event history is currently unavailable.'}
                </p>
              </div>
            ) : eventsData.events.length === 0 ? (
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
                  No lifecycle events recorded.
                </p>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  Legacy tickets created prior to event tracking have no entries in the lifecycle log; past transitions
                  are not synthesized from current ticket fields.
                </p>
              </div>
            ) : (
              <ol
                style={{
                  listStyle: 'none',
                  padding: 0,
                  margin: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '12px',
                }}
              >
                {eventsData.events.map((ev) => (
                  <li
                    key={ev.id}
                    style={{
                      border: '1px solid var(--border-color)',
                      borderRadius: '4px',
                      padding: '12px 14px',
                      backgroundColor: 'var(--bg-subtle)',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: '8px',
                        marginBottom: '6px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <strong style={{ fontSize: '13px', color: 'var(--text-main)' }}>{ev.label}</strong>
                        <span
                          className="badge"
                          style={{
                            fontSize: '11px',
                            textTransform: 'capitalize',
                            backgroundColor: 'var(--bg-card)',
                            border: '1px solid var(--border-color)',
                            color: 'var(--text-muted)',
                          }}
                        >
                          Actor: {ev.actor_type}
                        </span>
                      </div>
                      <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}>
                        Occurred: {formatUtcTimestamp(ev.occurred_at)}
                      </span>
                    </div>

                    {ev.recorded_at && ev.recorded_at !== ev.occurred_at && (
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                        Recorded in event log: {formatUtcTimestamp(ev.recorded_at)}
                      </div>
                    )}

                    {!ev.schema_supported ? (
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        Event schema version v{ev.schema_version} is unsupported in this UI; details omitted.
                      </div>
                    ) : !ev.payload_valid || !ev.details ? (
                      <div style={{ fontSize: '12px', color: 'var(--danger)', fontStyle: 'italic' }}>
                        Recorded event payload is malformed or incomplete for schema v{ev.schema_version}; details
                        omitted.
                      </div>
                    ) : (
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        {ev.event_type === 'ticket_created' && (
                          <>
                            {(ev.details.priority || ev.details.category) && (
                              <div>
                                Intake fields:{' '}
                                {ev.details.priority && (
                                  <span>
                                    Priority <strong style={{ color: 'var(--text-main)' }}>{ev.details.priority}</strong>
                                  </span>
                                )}
                                {ev.details.priority && ev.details.category ? ' · ' : ''}
                                {ev.details.category && (
                                  <span>
                                    Category <strong style={{ color: 'var(--text-main)' }}>{ev.details.category}</strong>
                                  </span>
                                )}
                              </div>
                            )}
                            {ev.details.intake_policy && (
                              <div>
                                <strong style={{ color: 'var(--text-main)' }}>Policy at intake:</strong>{' '}
                                {ev.details.intake_policy.is_configured ? (
                                  <span>
                                    {ev.details.intake_policy.policy_priority || 'default'} policy
                                    {ev.details.intake_policy.first_response_minutes !== null &&
                                      ` · First response ${formatDuration(ev.details.intake_policy.first_response_minutes)}`}
                                    {ev.details.intake_policy.resolution_minutes !== null &&
                                      ` · Resolution ${formatDuration(ev.details.intake_policy.resolution_minutes)}`}
                                    {' (snapshot at creation; does not freeze active targets or reflect later edits).'}
                                  </span>
                                ) : (
                                  <span>Unconfigured at ticket creation.</span>
                                )}
                              </div>
                            )}
                          </>
                        )}

                        {ev.event_type === 'ticket_processed' && (
                          <div>
                            {ev.details.from_status && ev.details.to_status && (
                              <span>
                                Transition:{' '}
                                <strong style={{ color: 'var(--text-main)' }}>
                                  {ev.details.from_status} → {ev.details.to_status}
                                </strong>
                              </span>
                            )}
                            {ev.details.predicted_category && (
                              <span>
                                {ev.details.from_status ? ' · ' : ''}
                                Predicted category:{' '}
                                <strong style={{ color: 'var(--text-main)' }}>{ev.details.predicted_category}</strong>
                              </span>
                            )}
                          </div>
                        )}

                        {ev.event_type === 'first_response_recorded' &&
                          ev.details.elapsed_calendar_seconds !== null && (
                            <div>
                              Elapsed calendar time since creation:{' '}
                              <strong style={{ color: 'var(--text-main)' }}>
                                {formatElapsedSeconds(ev.details.elapsed_calendar_seconds)}
                              </strong>
                            </div>
                          )}

                        {ev.event_type === 'ticket_resolved' && (
                          <div>
                            {ev.details.from_status && ev.details.to_status && (
                              <span>
                                Transition:{' '}
                                <strong style={{ color: 'var(--text-main)' }}>
                                  {ev.details.from_status} → {ev.details.to_status}
                                </strong>
                              </span>
                            )}
                            {ev.details.elapsed_calendar_seconds !== null && (
                              <span>
                                {ev.details.from_status ? ' · ' : ''}
                                Elapsed calendar time:{' '}
                                <strong style={{ color: 'var(--text-main)' }}>
                                  {formatElapsedSeconds(ev.details.elapsed_calendar_seconds)}
                                </strong>
                              </span>
                            )}
                          </div>
                        )}

                        {ev.event_type === 'ticket_closed' &&
                          ev.details.from_status &&
                          ev.details.to_status && (
                            <div>
                              Transition:{' '}
                              <strong style={{ color: 'var(--text-main)' }}>
                                {ev.details.from_status} → {ev.details.to_status}
                              </strong>
                            </div>
                          )}
                      </div>
                    )}
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>

        {/* Right: Triage & Model Diagnostics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Card: SLA Target & Timeliness */}
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
                  margin: 0,
                }}
              >
                SLA Target & Timeliness
              </h2>
              {slaData?.schema_available && slaData?.is_configured && (
                <div style={{ display: 'flex', gap: '6px' }}>
                  <span
                    className={`badge ${
                      slaData.overall_status === 'breached'
                        ? 'badge-failed'
                        : slaData.overall_status === 'fulfilled'
                        ? 'badge-ready'
                        : 'badge-pred'
                    }`}
                    style={{ textTransform: 'uppercase' }}
                  >
                    {slaData.overall_status}
                  </span>
                  {slaData.overall_urgency && (
                    <span
                      className={`badge ${
                        slaData.overall_urgency === 'overdue'
                          ? 'badge-failed'
                          : slaData.overall_urgency === 'high'
                          ? 'badge-pending'
                          : 'badge-ready'
                      }`}
                      style={{ textTransform: 'capitalize' }}
                    >
                      {slaData.overall_urgency} Urgency
                    </span>
                  )}
                </div>
              )}
            </div>

            {slaLoading ? (
              <div style={{ padding: '16px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                Evaluating SLA targets...
              </div>
            ) : slaError ? (
              <div className="alert alert-danger" style={{ margin: 0, fontSize: '12px' }}>
                {slaError}
              </div>
            ) : !slaData?.schema_available ? (
              <div
                style={{
                  backgroundColor: 'var(--bg-subtle)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '4px',
                  padding: '12px 14px',
                }}
              >
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
                  SLA Schema Unavailable
                </div>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  Migration 008 is not provisioned in the database. SLA target evaluation and policy enforcement are
                  inactive.
                </p>
              </div>
            ) : !slaData?.is_configured ? (
              <div
                style={{
                  backgroundColor: 'var(--bg-subtle)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '4px',
                  padding: '12px 14px',
                }}
              >
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
                  No SLA Policy Configured
                </div>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  No matching policy for priority <code>{ticket.priority || 'default'}</code> or tenant default.
                  Configure targets via the Admin SLA Policies menu.
                </p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13px' }}>
                {/* First Response Row */}
                {slaData.first_response && (
                  <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '4px',
                      }}
                    >
                      <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>First Response Target</span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
                        {formatDuration(slaData.first_response.target_minutes)}
                      </span>
                    </div>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        fontSize: '12px',
                      }}
                    >
                      <span style={{ color: 'var(--text-muted)' }}>
                        Due: {new Date(slaData.first_response.due_at).toISOString().replace('T', ' ').slice(0, 16)} UTC
                      </span>
                      <span
                        style={{
                          fontWeight: 600,
                          color:
                            slaData.first_response.timeliness_label === 'Due now' ||
                            slaData.first_response.is_breached
                              ? 'var(--danger)'
                              : slaData.first_response.is_completed
                              ? '#137333'
                              : 'var(--primary)',
                        }}
                      >
                        {slaData.first_response.timeliness_label ||
                          (slaData.first_response.is_completed ? 'Fulfilled' : 'On track')}
                      </span>
                    </div>
                    {slaData.first_response.completed_at && (
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                        Completed:{' '}
                        {new Date(slaData.first_response.completed_at).toISOString().replace('T', ' ').slice(0, 16)} UTC
                      </div>
                    )}
                  </div>
                )}

                {/* Resolution Row */}
                {slaData.resolution && (
                  <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '4px',
                      }}
                    >
                      <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>Resolution Target</span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
                        {formatDuration(slaData.resolution.target_minutes)}
                      </span>
                    </div>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        fontSize: '12px',
                      }}
                    >
                      <span style={{ color: 'var(--text-muted)' }}>
                        Due: {new Date(slaData.resolution.due_at).toISOString().replace('T', ' ').slice(0, 16)} UTC
                      </span>
                      <span
                        style={{
                          fontWeight: 600,
                          color:
                            slaData.resolution.timeliness_label === 'Due now' ||
                            slaData.resolution.is_breached
                              ? 'var(--danger)'
                              : slaData.resolution.is_completed
                              ? '#137333'
                              : 'var(--primary)',
                        }}
                      >
                        {slaData.resolution.timeliness_label ||
                          (slaData.resolution.is_completed ? 'Fulfilled' : 'On track')}
                      </span>
                    </div>
                    {slaData.resolution.completed_at && (
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                        Resolved:{' '}
                        {new Date(slaData.resolution.completed_at).toISOString().replace('T', ' ').slice(0, 16)} UTC
                      </div>
                    )}
                  </div>
                )}

                {/* Notice */}
                <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                  <strong>Notice:</strong> {slaData.urgency_notice}
                </p>
              </div>
            )}
          </div>

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
