import { useState, FormEvent } from 'react';
import { api } from '../api';

interface CreateTicketViewProps {
  onSuccess: (ticketId: string) => void;
  onCancel: () => void;
}

export const CreateTicketView = ({ onSuccess, onCancel }: CreateTicketViewProps) => {
  const [subject, setSubject] = useState('');
  const [body, setBody] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (submitting) return;

    const trimmedSubject = subject.trim();
    const trimmedBody = body.trim();

    if (!trimmedSubject) {
      setError('Subject must not be empty.');
      return;
    }
    if (!trimmedBody) {
      setError('Complaint body must not be empty.');
      return;
    }

    setSubmitting(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const res = await api.createTicket(trimmedSubject, trimmedBody);
      const newId = res.id || res.ticket?.id;
      setSuccessMessage('Ticket submitted successfully. Redirecting to detail view...');
      onSuccess(newId);
    } catch (err: unknown) {
      setSubmitting(false);
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred while submitting the ticket.');
      }
    }
  };

  return (
    <div style={{ maxWidth: '780px', margin: '0 auto' }}>
      <div style={{ marginBottom: '20px' }}>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onCancel}
          disabled={submitting}
          style={{ fontSize: '13px' }}
        >
          ← Back to Ticket Queue
        </button>
      </div>

      <div className="card" style={{ padding: '28px' }}>
        <div style={{ marginBottom: '20px' }}>
          <h1 style={{ fontSize: '20px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
            Submit Financial Support Ticket
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Submit a new customer complaint narrative. The background classifier will automatically triage and categorize the financial product.
          </p>
        </div>

        <div
          style={{
            backgroundColor: 'var(--warning-bg)',
            border: '1px solid var(--warning-border)',
            color: 'var(--warning-text)',
            padding: '12px 16px',
            borderRadius: '4px',
            fontSize: '13px',
            marginBottom: '20px',
            lineHeight: 1.4,
          }}
        >
          <strong>Domain Notice:</strong> This platform is powered by the CFPB-trained financial product classifier (run_004). It does not detect out-of-domain inquiries (e.g. IT support, generic bug reports).
        </div>

        {error && (
          <div className="alert alert-danger" role="alert">
            {error}
          </div>
        )}

        {successMessage && (
          <div className="alert alert-success" role="status">
            {successMessage}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <label htmlFor="ticket-subject" className="form-label" style={{ marginBottom: 0 }}>
                Subject
              </label>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                {subject.length}/300
              </span>
            </div>
            <input
              id="ticket-subject"
              type="text"
              className="form-input"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="e.g. Dispute over uncredited deposit and overdraft fees"
              maxLength={300}
              required
              disabled={submitting}
              autoFocus
            />
          </div>

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <label htmlFor="ticket-body" className="form-label" style={{ marginBottom: 0 }}>
                Complaint Narrative Body
              </label>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                {body.length}/20,000
              </span>
            </div>
            <textarea
              id="ticket-body"
              className="form-input"
              value={body}
              onChange={(e) => setBody(e.target.value)}
              placeholder="Provide the complete financial complaint narrative..."
              rows={8}
              maxLength={20000}
              required
              disabled={submitting}
              style={{ resize: 'vertical', lineHeight: 1.5 }}
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '24px' }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onCancel}
              disabled={submitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={submitting || !subject.trim() || !body.trim()}
            >
              {submitting ? 'Submitting Ticket...' : 'Submit Ticket'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

