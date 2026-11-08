import { useEffect, useState } from 'react';
import { api } from '../api';
import { SLAPolicyItem } from '../types';

interface SlaPoliciesModalProps {
  onClose: () => void;
  onPolicyChanged?: () => void;
}

const VALID_PRIORITIES = ['urgent', 'high', 'normal', 'low', 'default'] as const;

export const SlaPoliciesModal = ({ onClose, onPolicyChanged }: SlaPoliciesModalProps) => {
  const [policies, setPolicies] = useState<SLAPolicyItem[]>([]);
  const [schemaAvailable, setSchemaAvailable] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [deletingPriority, setDeletingPriority] = useState<string | null>(null);
  const [confirmDeletePriority, setConfirmDeletePriority] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Form state
  const [selectedPriority, setSelectedPriority] = useState<string>('normal');
  const [firstResponseMinutes, setFirstResponseMinutes] = useState<number>(60);
  const [resolutionMinutes, setResolutionMinutes] = useState<number>(240);

  const fetchPolicies = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getSlaPolicies();
      setSchemaAvailable(res.schema_available);
      setPolicies(res.policies || []);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch SLA policies.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPolicies();
  }, []);

  // Handle ESC key to close modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!schemaAvailable) {
      setError('Cannot save policy: SLA schema is not provisioned (migration 008 unapplied).');
      return;
    }
    if (firstResponseMinutes <= 0 || resolutionMinutes <= 0) {
      setError('Target durations must be positive integers.');
      return;
    }
    setSaving(true);
    setError(null);
    setSuccessMsg(null);
    try {
      const res = await api.upsertSlaPolicy(selectedPriority, firstResponseMinutes, resolutionMinutes);
      setSuccessMsg(`Policy for '${res.policy.priority}' saved successfully.`);
      onPolicyChanged?.();
      await fetchPolicies();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to save SLA policy.');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (priority: string) => {
    setDeletingPriority(priority);
    setError(null);
    setSuccessMsg(null);
    try {
      await api.deleteSlaPolicy(priority);
      setSuccessMsg(`Policy for '${priority}' deleted.`);
      setConfirmDeletePriority(null);
      onPolicyChanged?.();
      await fetchPolicies();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : `Failed to delete policy '${priority}'.`);
    } finally {
      setDeletingPriority(null);
    }
  };

  const formatMinutes = (mins: number) => {
    if (mins < 60) return `${mins}m`;
    const hours = (mins / 60).toFixed(1).replace('.0', '');
    if (mins < 1440) return `${mins}m (${hours}h)`;
    const days = (mins / 1440).toFixed(1).replace('.0', '');
    return `${mins}m (${days}d)`;
  };

  const getPriorityBadgeStyle = (priority: string): React.CSSProperties => {
    switch (priority) {
      case 'urgent':
        return { backgroundColor: '#fef2f2', color: '#991b1b', border: '1px solid #fca5a5' };
      case 'high':
        return { backgroundColor: '#fff7ed', color: '#9a3412', border: '1px solid #fed7aa' };
      case 'normal':
        return { backgroundColor: '#f1f5f9', color: '#334155', border: '1px solid #cbd5e1' };
      case 'low':
        return { backgroundColor: '#f8fafc', color: '#475569', border: '1px solid #e2e8f0' };
      case 'default':
        return { backgroundColor: '#eef2ff', color: '#3730a3', border: '1px solid #c7d2fe' };
      default:
        return { backgroundColor: '#f1f5f9', color: '#334155', border: '1px solid #cbd5e1' };
    }
  };

  return (
    <>
      {/* 1. Translucent backdrop behind the panel */}
      <div
        className="sla-modal-backdrop"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* 2. Modal Overlay & Solid Panel */}
      <div
        className="sla-modal-overlay"
        role="presentation"
        onClick={(e) => {
          if (e.target === e.currentTarget) onClose();
        }}
      >
        <div
          className="sla-modal-panel"
          role="dialog"
          aria-modal="true"
          aria-labelledby="sla-modal-title"
          aria-describedby="sla-modal-desc"
        >
          {/* Header */}
          <div className="sla-modal-header">
            <div>
              <h2
                id="sla-modal-title"
                style={{ fontSize: '18px', fontWeight: 700, color: '#0f172a', margin: 0 }}
              >
                Tenant SLA Policies
              </h2>
              <p
                id="sla-modal-desc"
                style={{ fontSize: '13px', color: '#475569', margin: '4px 0 0' }}
              >
                Deterministic response and resolution targets scoped to your tenant.
              </p>
            </div>
            <button
              type="button"
              onClick={onClose}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '24px',
                cursor: 'pointer',
                color: '#64748b',
                padding: '4px 8px',
                lineHeight: 1,
                borderRadius: '4px',
              }}
              aria-label="Close modal"
              title="Close modal (Esc)"
            >
              ×
            </button>
          </div>

          {/* Scrollable Content */}
          <div className="sla-modal-body">
            {!schemaAvailable && (
              <div
                style={{
                  backgroundColor: '#fffbeb',
                  color: '#92400e',
                  border: '1px solid #fde68a',
                  padding: '12px 16px',
                  borderRadius: '6px',
                  fontSize: '13px',
                  lineHeight: 1.5,
                }}
              >
                <strong style={{ color: '#78350f' }}>Schema Unavailable:</strong> Migration 008 (<code>sla_policies</code>) is not provisioned in the
                active database. Policy writes will return 503 until the migration is executed.
              </div>
            )}

            {error && (
              <div className="alert alert-danger" style={{ margin: 0, fontSize: '13px' }}>
                {error}
              </div>
            )}

            {successMsg && (
              <div className="alert alert-success" style={{ margin: 0, fontSize: '13px' }}>
                {successMsg}
              </div>
            )}

            {/* Existing Policies List */}
            <div>
              <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#0f172a', marginBottom: '10px' }}>
                Configured Policies ({policies.length})
              </h3>

              {loading ? (
                <div style={{ padding: '24px', textAlign: 'center', color: '#64748b', fontSize: '13px' }}>
                  Loading policies...
                </div>
              ) : policies.length === 0 ? (
                <div
                  style={{
                    backgroundColor: '#f8fafc',
                    padding: '20px',
                    borderRadius: '6px',
                    textAlign: 'center',
                    fontSize: '13px',
                    color: '#475569',
                    border: '1px solid #e2e8f0',
                  }}
                >
                  No SLA policies configured for this tenant. Add a policy below.
                </div>
              ) : (
                <div style={{ overflowX: 'auto', border: '1px solid #e2e8f0', borderRadius: '6px' }}>
                  <table className="sla-modal-table">
                    <thead>
                      <tr>
                        <th>Priority Key</th>
                        <th>First Response</th>
                        <th>Resolution</th>
                        <th style={{ textAlign: 'right' }}>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {policies.map((p) => {
                        const badgeStyle = getPriorityBadgeStyle(p.priority);
                        return (
                          <tr key={p.id}>
                            <td>
                              <span
                                style={{
                                  fontWeight: 600,
                                  textTransform: 'uppercase',
                                  fontSize: '11px',
                                  letterSpacing: '0.5px',
                                  padding: '3px 8px',
                                  borderRadius: '4px',
                                  display: 'inline-block',
                                  ...badgeStyle,
                                }}
                              >
                                {p.priority} {p.priority === 'default' ? '(Tenant Default)' : ''}
                              </span>
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 500, color: '#1e293b' }}>
                              {formatMinutes(p.first_response_minutes)}
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 500, color: '#1e293b' }}>
                              {formatMinutes(p.resolution_minutes)}
                            </td>
                            <td style={{ textAlign: 'right' }}>
                              {confirmDeletePriority === p.priority ? (
                                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                                  <span style={{ fontSize: '12px', fontWeight: 600, color: '#b91c1c' }}>Delete?</span>
                                  <button
                                    type="button"
                                    disabled={deletingPriority === p.priority}
                                    onClick={() => handleDelete(p.priority)}
                                    style={{
                                      padding: '4px 10px',
                                      fontSize: '12px',
                                      fontWeight: 600,
                                      backgroundColor: '#dc2626',
                                      color: '#ffffff',
                                      border: 'none',
                                      borderRadius: '4px',
                                      cursor: 'pointer',
                                    }}
                                  >
                                    {deletingPriority === p.priority ? '...' : 'Yes, Delete'}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => setConfirmDeletePriority(null)}
                                    style={{
                                      padding: '4px 8px',
                                      fontSize: '12px',
                                      background: '#ffffff',
                                      border: '1px solid #cbd5e1',
                                      borderRadius: '4px',
                                      color: '#334155',
                                      cursor: 'pointer',
                                    }}
                                  >
                                    Cancel
                                  </button>
                                </div>
                              ) : (
                                <button
                                  type="button"
                                  disabled={!schemaAvailable}
                                  onClick={() => setConfirmDeletePriority(p.priority)}
                                  style={{
                                    padding: '4px 10px',
                                    fontSize: '12px',
                                    fontWeight: 500,
                                    background: '#ffffff',
                                    border: '1px solid #fca5a5',
                                    borderRadius: '4px',
                                    color: '#b91c1c',
                                    cursor: schemaAvailable ? 'pointer' : 'not-allowed',
                                  }}
                                >
                                  Delete
                                </button>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Add / Update Policy Form */}
            <div
              style={{
                borderTop: '1px solid #e2e8f0',
                paddingTop: '18px',
              }}
            >
              <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#0f172a', marginBottom: '12px' }}>
                Add or Update Policy
              </h3>
              <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '14px' }}>
                  <div>
                    <label
                      htmlFor="sla-priority"
                      style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#1e293b', marginBottom: '6px' }}
                    >
                      Priority Key
                    </label>
                    <select
                      id="sla-priority"
                      value={selectedPriority}
                      onChange={(e) => setSelectedPriority(e.target.value)}
                      className="sla-modal-input"
                    >
                      {VALID_PRIORITIES.map((p) => (
                        <option key={p} value={p}>
                          {p.toUpperCase()} {p === 'default' ? '(Tenant Default)' : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label
                      htmlFor="sla-fr-target"
                      style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#1e293b', marginBottom: '6px' }}
                    >
                      First Response (mins)
                    </label>
                    <input
                      id="sla-fr-target"
                      type="number"
                      min={1}
                      max={525600}
                      value={firstResponseMinutes}
                      onChange={(e) => setFirstResponseMinutes(parseInt(e.target.value, 10) || 0)}
                      className="sla-modal-input"
                    />
                    <span style={{ fontSize: '12px', fontWeight: 500, color: '#475569', marginTop: '4px', display: 'block' }}>
                      ≈ {formatMinutes(firstResponseMinutes)}
                    </span>
                  </div>

                  <div>
                    <label
                      htmlFor="sla-res-target"
                      style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#1e293b', marginBottom: '6px' }}
                    >
                      Resolution (mins)
                    </label>
                    <input
                      id="sla-res-target"
                      type="number"
                      min={1}
                      max={525600}
                      value={resolutionMinutes}
                      onChange={(e) => setResolutionMinutes(parseInt(e.target.value, 10) || 0)}
                      className="sla-modal-input"
                    />
                    <span style={{ fontSize: '12px', fontWeight: 500, color: '#475569', marginTop: '4px', display: 'block' }}>
                      ≈ {formatMinutes(resolutionMinutes)}
                    </span>
                  </div>
                </div>

                <div
                  style={{
                    backgroundColor: '#f8fafc',
                    border: '1px solid #e2e8f0',
                    borderRadius: '6px',
                    padding: '10px 14px',
                    fontSize: '12px',
                    color: '#334155',
                    lineHeight: 1.5,
                  }}
                >
                  <strong style={{ color: '#0f172a' }}>Notice:</strong> Policy targets are evaluated deterministically against elapsed calendar time
                  using UTC timestamps. Saving or deleting a policy takes immediate effect on all tickets in this tenant.
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '4px' }}>
                  <button
                    type="button"
                    onClick={onClose}
                    style={{
                      padding: '8px 16px',
                      fontSize: '13px',
                      fontWeight: 500,
                      backgroundColor: '#ffffff',
                      color: '#334155',
                      border: '1px solid #cbd5e1',
                      borderRadius: '4px',
                      cursor: 'pointer',
                    }}
                  >
                    Close
                  </button>
                  <button
                    type="submit"
                    disabled={saving || !schemaAvailable}
                    style={{
                      padding: '8px 18px',
                      fontSize: '13px',
                      fontWeight: 600,
                      backgroundColor: saving || !schemaAvailable ? '#94a3b8' : '#1e3a8a',
                      color: '#ffffff',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: saving || !schemaAvailable ? 'not-allowed' : 'pointer',
                    }}
                  >
                    {saving ? 'Saving...' : 'Save Policy'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};
