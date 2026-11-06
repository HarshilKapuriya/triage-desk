import { TicketListItem, Pagination } from '../types';

interface TicketsListViewProps {
  tickets: TicketListItem[];
  pagination: Pagination;
  loading: boolean;
  onPageChange: (newPage: number) => void;
  onSelectTicket: (ticketId: string) => void;
  onNewTicket: () => void;
}

export const TicketsListView = ({
  tickets,
  pagination,
  loading,
  onPageChange,
  onSelectTicket,
  onNewTicket,
}: TicketsListViewProps) => {
  const formatDate = (dateStr: string | null): string => {
    if (!dateStr) return '—';
    try {
      const d = new Date(dateStr);
      return d.toISOString().replace('T', ' ').slice(0, 16);
    } catch {
      return dateStr;
    }
  };

  return (
    <div>
      <div
        style={{
          marginBottom: '24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end',
        }}
      >
        <div>
          <h1 style={{ fontSize: '22px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
            Ticket Queue
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Review tenant customer complaints, human-assigned categories, and automated CFPB product predictions.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Total Tickets: <strong style={{ color: 'var(--text-main)' }}>{pagination.total_count}</strong>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            onClick={onNewTicket}
            style={{ padding: '7px 14px', fontSize: '13px', fontWeight: 600 }}
          >
            + New Ticket
          </button>
        </div>
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '140px' }}>Ticket ID</th>
                <th style={{ width: '140px' }}>Received</th>
                <th>Subject & Customer</th>
                <th style={{ width: '90px' }}>Status</th>
                <th style={{ width: '150px' }}>Human Category</th>
                <th style={{ width: '270px' }}>Model Prediction</th>
                <th style={{ width: '80px', textAlign: 'center' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {tickets.length > 0 ? (
                tickets.map((t) => (
                  <tr key={t.id}>
                    <td>
                      <span className="text-mono" title={t.id} style={{ fontWeight: 500 }}>
                        {t.external_id || `${t.id.slice(0, 8)}...`}
                      </span>
                    </td>
                    <td style={{ color: 'var(--text-muted)', fontSize: '12px', whiteSpace: 'nowrap' }}>
                      {formatDate(t.created_at)}
                    </td>
                    <td>
                      <div style={{ fontWeight: 500, color: 'var(--text-main)', marginBottom: '2px' }}>
                        <button
                          type="button"
                          onClick={() => onSelectTicket(t.id)}
                          style={{
                            background: 'none',
                            border: 'none',
                            padding: 0,
                            font: 'inherit',
                            fontWeight: 500,
                            color: 'inherit',
                            cursor: 'pointer',
                            textAlign: 'left',
                          }}
                        >
                          {t.subject}
                        </button>
                      </div>
                      {t.customer_email && (
                        <div className="text-muted" style={{ fontSize: '12px' }}>
                          {t.customer_email}
                        </div>
                      )}
                    </td>
                    <td>
                      <span className={`badge badge-status-${t.status.toLowerCase()}`}>{t.status}</span>
                    </td>
                    <td>
                      {t.category ? (
                        <span style={{ fontSize: '12px', color: 'var(--text-main)', fontWeight: 500 }}>
                          {t.category}
                        </span>
                      ) : (
                        <span className="text-muted" style={{ fontSize: '12px', fontStyle: 'italic' }}>
                          Unassigned
                        </span>
                      )}
                    </td>
                    <td>
                      {t.predicted_label ? (
                        <div>
                          <div style={{ marginBottom: '2px' }}>
                            <span className="badge badge-pred">{t.predicted_label}</span>
                          </div>
                          <div className="text-mono text-muted" style={{ fontSize: '11px' }}>
                            {t.model_name} ({t.model_version})
                          </div>
                        </div>
                      ) : (
                        <span className="badge badge-pending">Prediction Pending</span>
                      )}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{ padding: '4px 10px', fontSize: '12px' }}
                        onClick={() => onSelectTicket(t.id)}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '48px 16px', color: 'var(--text-muted)' }}>
                    {loading ? 'Loading tickets...' : 'No tickets found in this tenant inbox.'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {pagination.total_pages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '4px 0' }}>
          <div>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={pagination.page <= 1 || loading}
              onClick={() => onPageChange(pagination.page - 1)}
            >
              Previous
            </button>
          </div>

          <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Page {pagination.page} of {pagination.total_pages}
          </div>

          <div>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={pagination.page >= pagination.total_pages || loading}
              onClick={() => onPageChange(pagination.page + 1)}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

