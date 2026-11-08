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

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: '20px',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="card"
        style={{
          width: '100%',
          maxWidth: '720px',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'var(--bg-card)',
          borderRadius: '8px',
          boxShadow: '0 8px 30px rgba(0, 0, 0, 0.2)',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '16px 24px',
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-main)', margin: 0 }}>
              Tenant SLA Policies
            </h2>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '4px 0 0' }}>
              Deterministic response and resolution targets scoped to your tenant.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '20px',
              cursor: 'pointer',
              color: 'var(--text-muted)',
              lineHeight: 1,
            }}
            aria-label="Close modal"
          >
            ×
          </button>
        </div>

        {/* Scrollable Content */}
        <div style={{ padding: '24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {!schemaAvailable && (
            <div
              style={{
                backgroundColor: '#fff3cd',
                color: '#856404',
                border: '1px solid #ffeeba',
                padding: '12px 16px',
                borderRadius: '4px',
                fontSize: '13px',
                lineHeight: 1.5,
              }}
            >
              <strong>Schema Unavailable:</strong> Migration 008 (<code>sla_policies</code>) is not provisioned in the
              active database. Policy writes will return 503 until the migration is executed.
            </div>
          )}

          {error && (
            <div className="alert alert-danger" style={{ margin: 0, fontSize: '13px' }}>
              {error}
            </div>
          )}

          {successMsg && (
            <div
              style={{
                backgroundColor: '#d4edda',
                color: '#155724',
                border: '1px solid #c3e6cb',
                padding: '10px 14px',
                borderRadius: '4px',
                fontSize: '13px',
              }}
            >
              {successMsg}
            </div>
          )}

          {/* Existing Policies List */}
          <div>
            <h3 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '8px' }}>
              Configured Policies ({policies.length})
            </h3>

            {loading ? (
              <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                Loading policies...
              </div>
            ) : policies.length === 0 ? (
              <div
                style={{
                  backgroundColor: 'var(--bg-subtle)',
                  padding: '16px',
                  borderRadius: '4px',
                  textAlign: 'center',
                  fontSize: '13px',
                  color: 'var(--text-muted)',
                  border: '1px solid var(--border-color)',
                }}
              >
                No SLA policies configured for this tenant. Add a policy below.
              </div>
            ) : (
              <div style={{ overflowX: 'auto', border: '1px solid var(--border-color)', borderRadius: '4px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ backgroundColor: 'var(--bg-subtle)', textAlign: 'left', borderBottom: '1px solid var(--border-color)' }}>
                      <th style={{ padding: '8px 12px', fontWeight: 600 }}>Priority Key</th>
                      <th style={{ padding: '8px 12px', fontWeight: 600 }}>First Response</th>
                      <th style={{ padding: '8px 12px', fontWeight: 600 }}>Resolution</th>
                      <th style={{ padding: '8px 12px', fontWeight: 600, textAlign: 'right' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {policies.map((p) => (
                      <tr key={p.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <td style={{ padding: '8px 12px' }}>
                          <span
                            style={{
                              fontWeight: 600,
                              textTransform: 'uppercase',
                              fontSize: '11px',
                              letterSpacing: '0.5px',
                              padding: '2px 6px',
                              borderRadius: '3px',
                              backgroundColor: p.priority === 'default' ? '#e2e8f0' : 'var(--bg-subtle)',
                              color: 'var(--text-main)',
                            }}
                          >
                            {p.priority}
                          </span>
                        </td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>
                          {formatMinutes(p.first_response_minutes)}
                        </td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>
                          {formatMinutes(p.resolution_minutes)}
                        </td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                          {confirmDeletePriority === p.priority ? (
                            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                              <span style={{ fontSize: '11px', color: 'var(--danger)' }}>Confirm?</span>
                              <button
                                type="button"
                                disabled={deletingPriority === p.priority}
                                onClick={() => handleDelete(p.priority)}
                                style={{
                                  padding: '2px 8px',
                                  fontSize: '11px',
                                  backgroundColor: 'var(--danger)',
                                  color: '#fff',
                                  border: 'none',
                                  borderRadius: '3px',
                                  cursor: 'pointer',
                                }}
                              >
                                {deletingPriority === p.priority ? '...' : 'Yes, Delete'}
                              </button>
                              <button
                                type="button"
                                onClick={() => setConfirmDeletePriority(null)}
                                style={{
                                  padding: '2px 6px',
                                  fontSize: '11px',
                                  background: 'none',
                                  border: '1px solid var(--border-color)',
                                  borderRadius: '3px',
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
                                padding: '2px 8px',
                                fontSize: '12px',
                                background: 'none',
                                border: '1px solid var(--border-color)',
                                borderRadius: '3px',
                                color: 'var(--danger)',
                                cursor: schemaAvailable ? 'pointer' : 'not-allowed',
                              }}
                            >
                              Delete
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Add / Update Policy Form */}
          <div
            style={{
              borderTop: '1px solid var(--border-color)',
              paddingTop: '16px',
            }}
          >
            <h3 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '12px' }}>
              Add or Update Policy
            </h3>
            <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                <div>
                  <label
                    htmlFor="sla-priority"
                    style={{ display: 'block', fontSize: '12px', fontWeight: 600, marginBottom: '4px' }}
                  >
                    Priority Key
                  </label>
                  <select
                    id="sla-priority"
                    value={selectedPriority}
                    onChange={(e) => setSelectedPriority(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid var(--border-color)',
                      fontSize: '13px',
                      backgroundColor: 'var(--bg-main)',
                      color: 'var(--text-main)',
                    }}
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
                    style={{ display: 'block', fontSize: '12px', fontWeight: 600, marginBottom: '4px' }}
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
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid var(--border-color)',
                      fontSize: '13px',
                    }}
                  />
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px', display: 'block' }}>
                    ≈ {formatMinutes(firstResponseMinutes)}
                  </span>
                </div>

                <div>
                  <label
                    htmlFor="sla-res-target"
                    style={{ display: 'block', fontSize: '12px', fontWeight: 600, marginBottom: '4px' }}
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
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid var(--border-color)',
                      fontSize: '13px',
                    }}
                  />
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px', display: 'block' }}>
                    ≈ {formatMinutes(resolutionMinutes)}
                  </span>
                </div>
              </div>

              <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
                <strong>Notice:</strong> Policy targets are evaluated deterministically against elapsed calendar time
                using UTC timestamps. Saving or deleting a policy takes immediate effect on all tickets in this tenant.
              </p>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '6px' }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={onClose}
                  style={{ padding: '6px 14px', fontSize: '13px' }}
                >
                  Close
                </button>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={saving || !schemaAvailable}
                  style={{ padding: '6px 16px', fontSize: '13px' }}
                >
                  {saving ? 'Saving...' : 'Save Policy'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};
