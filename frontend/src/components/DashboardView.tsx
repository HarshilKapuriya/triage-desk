import React, { useEffect, useState, useRef, useCallback } from 'react';
import { api } from '../api';
import { User, DashboardResponse, DashboardTargetMetrics } from '../types';

interface DashboardViewProps {
  user: User | null;
  onOpenSlaModal?: () => void;
  slaPolicyVersion?: number;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  user,
  onOpenSlaModal,
  slaPolicyVersion = 0,
}) => {
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef<number>(0);

  const fetchDashboard = useCallback(async () => {
    // Abort previous in-flight request if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const currentRequestId = ++requestIdRef.current;

    setLoading(true);
    setError(null);

    try {
      const res = await api.getDashboard(controller.signal);
      // Strictly ignore stale responses if another request was initiated
      if (currentRequestId === requestIdRef.current) {
        setData(res);
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        // Ignored aborted request
        return;
      }
      if (currentRequestId === requestIdRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard metrics.');
      }
    } finally {
      if (currentRequestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    fetchDashboard();
    return () => {
      // Invalidate in-flight requests and abort controller on unmount or version change
      requestIdRef.current++;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchDashboard, slaPolicyVersion]);

  const formatRate = (rate: number | null): string => {
    if (rate === null || rate === undefined) {
      return 'Not available';
    }
    return `${(rate * 100).toFixed(1)}%`;
  };

  const renderTargetCard = (
    title: string,
    target: DashboardTargetMetrics,
    unconfiguredCount: number
  ) => {
    return (
      <div className="dashboard-target-card">
        <div className="dashboard-target-header">
          <h3 className="dashboard-target-title">{title}</h3>
          <span className="dashboard-target-badge">
            {target.eligible_total} eligible
            {unconfiguredCount > 0 && ` · ${unconfiguredCount} unconfigured`}
          </span>
        </div>

        {target.data_quality_issues > 0 && (
          <div className="dashboard-dq-warning">
            ⚠️ <strong>{target.data_quality_issues}</strong> ticket
            {target.data_quality_issues > 1 ? 's have' : ' has'} invalid timestamp data
            and {target.data_quality_issues > 1 ? 'are' : 'is'} excluded from rates.
          </div>
        )}

        <div className="dashboard-metric-section">
          <div className="dashboard-metric-row">
            <div>
              <div className="dashboard-metric-label">Completed On-Time Rate</div>
              <div className="dashboard-metric-sublabel">
                {target.completed_total > 0
                  ? `${target.completed_on_time} on-time of ${target.completed_total} completed`
                  : '0 completed tickets'}
              </div>
            </div>
            <div className={`dashboard-metric-value ${target.completed_timeliness_rate !== null && target.completed_timeliness_rate >= 0.8 ? 'positive' : ''}`}>
              {formatRate(target.completed_timeliness_rate)}
            </div>
          </div>
          <div className="dashboard-breakdown-chips">
            <span className="dashboard-chip chip-success">
              On-time: <strong>{target.completed_on_time}</strong>
            </span>
            <span className="dashboard-chip chip-danger">
              Late: <strong>{target.completed_late}</strong>
            </span>
          </div>
        </div>

        <div className="dashboard-metric-section">
          <div className="dashboard-metric-row">
            <div>
              <div className="dashboard-metric-label">Active Not-Overdue Rate</div>
              <div className="dashboard-metric-sublabel">
                {target.incomplete_total > 0
                  ? `${target.incomplete_before_deadline + target.incomplete_due_now} not overdue of ${target.incomplete_total} active`
                  : '0 active tickets'}
              </div>
            </div>
            <div className={`dashboard-metric-value ${target.incomplete_not_overdue_rate !== null && target.incomplete_not_overdue_rate >= 0.8 ? 'positive' : ''}`}>
              {formatRate(target.incomplete_not_overdue_rate)}
            </div>
          </div>
          <div className="dashboard-breakdown-chips">
            <span className="dashboard-chip chip-info">
              Before deadline: <strong>{target.incomplete_before_deadline}</strong>
            </span>
            <span className="dashboard-chip chip-warning">
              Due now: <strong>{target.incomplete_due_now}</strong>
            </span>
            <span className="dashboard-chip chip-danger">
              Overdue: <strong>{target.incomplete_overdue}</strong>
            </span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="dashboard-container">
      {/* Top Header */}
      <div className="dashboard-header-bar">
        <div>
          <h2 className="dashboard-title">Tenant Operational Dashboard</h2>
          <div className="dashboard-subtitle">
            All-time operational workload and deterministic SLA target metrics.
          </div>
          {data?.evaluated_at && (
            <div className="dashboard-timestamp">
              Evaluated snapshot: <code>{new Date(data.evaluated_at).toLocaleString()}</code> (UTC:{' '}
              {data.evaluated_at})
            </div>
          )}
        </div>
        <div className="dashboard-actions">
          {user?.role === 'admin' && onOpenSlaModal && (
            <button
              type="button"
              className="dashboard-btn-secondary"
              onClick={onOpenSlaModal}
            >
              ⚙ Manage SLA Policies
            </button>
          )}
          <button
            type="button"
            className="dashboard-btn-primary"
            onClick={fetchDashboard}
            disabled={loading}
          >
            {loading ? 'Refreshing...' : '🔄 Refresh Metrics'}
          </button>
        </div>
      </div>

      {/* Schema Unavailable State */}
      {data && !data.schema_available && (
        <div className="dashboard-alert alert-warning">
          <strong>SLA Schema Notice:</strong> SLA policy schema is not provisioned (migration 008 unapplied).
          Ticket status counts and workload metrics below remain live and accurate.
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="dashboard-alert alert-danger">
          <strong>Error loading dashboard:</strong> {error}
          <div style={{ marginTop: '8px' }}>
            <button type="button" className="dashboard-btn-secondary" onClick={fetchDashboard}>
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Loading State without Data */}
      {loading && !data && (
        <div className="dashboard-loading-box">
          <div className="dashboard-spinner" />
          <div>Computing all-time tenant operational metrics...</div>
        </div>
      )}

      {/* Content State */}
      {data && (
        <>
          {/* Workload Summary Cards */}
          <div className="dashboard-workload-grid">
            <div className="dashboard-card card-accent-open">
              <div className="dashboard-card-label">Open Workload</div>
              <div className="dashboard-card-value">{data.workload.open}</div>
              <div className="dashboard-card-detail">
                Active tickets: <code>new ({data.status_counts.new})</code> + <code>open ({data.status_counts.open})</code> + <code>pending ({data.status_counts.pending})</code>
              </div>
            </div>

            <div className="dashboard-card card-accent-completed">
              <div className="dashboard-card-label">Completed Workload</div>
              <div className="dashboard-card-value">{data.workload.completed}</div>
              <div className="dashboard-card-detail">
                Historical: <code>resolved ({data.status_counts.resolved})</code> + <code>closed ({data.status_counts.closed})</code>
              </div>
            </div>

            <div className="dashboard-card card-accent-total">
              <div className="dashboard-card-label">Total Volume</div>
              <div className="dashboard-card-value">{data.status_counts.total}</div>
              <div className="dashboard-card-detail">
                {data.workload.scope_label}
              </div>
            </div>
          </div>

          {/* Status Breakdown Bar */}
          <div className="dashboard-status-bar">
            <span className="dashboard-status-title">Status Breakdown:</span>
            <span className="dashboard-status-item status-new">
              New: <strong>{data.status_counts.new}</strong>
            </span>
            <span className="dashboard-status-item status-open">
              Open: <strong>{data.status_counts.open}</strong>
            </span>
            <span className="dashboard-status-item status-pending">
              Pending: <strong>{data.status_counts.pending}</strong>
            </span>
            <span className="dashboard-status-item status-resolved">
              Resolved: <strong>{data.status_counts.resolved}</strong>
            </span>
            <span className="dashboard-status-item status-closed">
              Closed: <strong>{data.status_counts.closed}</strong>
            </span>
          </div>

          {/* Empty Tenant Notice */}
          {data.status_counts.total === 0 && (
            <div className="dashboard-empty-box">
              <div style={{ fontSize: '24px', marginBottom: '8px' }}>📭</div>
              <h3>No tickets recorded for this tenant</h3>
              <p>When tickets are ingested or created, their SLA metrics and status counts will appear here.</p>
            </div>
          )}

          {/* SLA Section */}
          {data.schema_available && data.sla_summary && (
            <div className="dashboard-sla-section">
              <div className="dashboard-section-header">
                <div>
                  <h3 className="dashboard-section-title">Service Level Agreement (SLA) Summaries</h3>
                  <div className="dashboard-section-sub">
                    Deterministic evaluations based on recorded <code>first_response_at</code> and <code>resolved_at</code> timestamps.
                  </div>
                </div>
                <div className="dashboard-policy-status">
                  Policies Configured: <strong>{data.sla_summary.policies_configured}</strong>
                  {data.sla_summary.policies_configured === 0 && (
                    <span className="badge-warning" style={{ marginLeft: '8px' }}>No policies active</span>
                  )}
                </div>
              </div>

              {/* No Policies Notice */}
              {data.sla_summary.policies_configured === 0 && (
                <div className="dashboard-alert alert-info">
                  <strong>No SLA Policies Configured:</strong> All tenant tickets are currently unconfigured for SLA tracking.
                  {user?.role === 'admin' && onOpenSlaModal && (
                    <button
                      type="button"
                      className="dashboard-btn-secondary"
                      style={{ marginLeft: '12px' }}
                      onClick={onOpenSlaModal}
                    >
                      Configure Policies
                    </button>
                  )}
                </div>
              )}

              {/* Partial Policy Coverage Notice */}
              {data.sla_summary.policies_configured > 0 && data.sla_summary.unconfigured_tickets > 0 && (
                <div className="dashboard-alert alert-warning">
                  <strong>Partial Policy Coverage:</strong> {data.sla_summary.unconfigured_tickets} ticket
                  {data.sla_summary.unconfigured_tickets > 1 ? 's have' : ' has'} no matching priority policy and no default policy.
                  These tickets are excluded from SLA compliance rates.
                </div>
              )}

              {/* Target Cards Grid */}
              <div className="dashboard-targets-grid">
                {renderTargetCard(
                  'First Response Target',
                  data.sla_summary.first_response,
                  data.sla_summary.unconfigured_tickets
                )}
                {renderTargetCard(
                  'Resolution Target',
                  data.sla_summary.resolution,
                  data.sla_summary.unconfigured_tickets
                )}
              </div>
            </div>
          )}

          {/* Urgency Notice */}
          <div className="dashboard-disclaimer">
            <strong>Operational Notice:</strong> {data.urgency_notice}
          </div>
        </>
      )}
    </div>
  );
};
