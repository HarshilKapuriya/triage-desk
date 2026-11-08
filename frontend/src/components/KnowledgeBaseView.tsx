import { useEffect, useState } from 'react';
import { api } from '../api';
import { KBArticleSummary } from '../types';

interface KnowledgeBaseViewProps {
  onBack: () => void;
}

export const KnowledgeBaseView = ({ onBack }: KnowledgeBaseViewProps) => {
  const [articles, setArticles] = useState<KBArticleSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modal / form state
  const [showCreate, setShowCreate] = useState(false);
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [successNotice, setSuccessNotice] = useState<string | null>(null);

  const loadArticles = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getKBArticles();
      setArticles(res.items);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load knowledge base articles.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadArticles();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;

    const trimmedTitle = title.trim();
    const trimmedBody = body.trim();

    if (!trimmedTitle) {
      setFormError('Article title is required.');
      return;
    }
    if (trimmedTitle.length > 300) {
      setFormError('Title must not exceed 300 characters.');
      return;
    }
    if (!trimmedBody) {
      setFormError('Article body is required.');
      return;
    }
    if (trimmedBody.length > 20000) {
      setFormError('Article body must not exceed 20,000 characters.');
      return;
    }

    setSubmitting(true);
    setFormError(null);

    try {
      const created = await api.createKBArticle(trimmedTitle, trimmedBody);
      setTitle('');
      setBody('');
      setShowCreate(false);
      setSuccessNotice(`Article "${created.title}" created with ${created.chunk_count} chunk(s).`);
      loadArticles();
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : 'Failed to create article.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      {/* Top Bar */}
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
          className="btn btn-primary"
          onClick={() => {
            setShowCreate(!showCreate);
            setFormError(null);
            setSuccessNotice(null);
          }}
          style={{ fontSize: '13px', padding: '6px 14px' }}
        >
          {showCreate ? 'Cancel' : '+ Add Article'}
        </button>
      </div>

      {/* Header */}
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '22px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
          Knowledge Base
        </h1>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: 0 }}>
          Tenant-owned reference articles and policy documents. Ingested articles are deterministically partitioned into
          passage chunks for cosine-similarity retrieval against support complaints.
        </p>
      </div>

      {/* Success Notification */}
      {successNotice && (
        <div className="alert alert-success" style={{ marginBottom: '20px', fontSize: '13px' }}>
          {successNotice}
        </div>
      )}

      {/* Add Article Form */}
      {showCreate && (
        <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
          <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '16px' }}>
            <h2 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-main)', margin: 0 }}>
              New Knowledge Base Article
            </h2>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px', marginBottom: 0 }}>
              Provide reference documentation in plain text. Text is split into overlapping passage chunks bounded
              within model token limits (max 256 subwords).
            </p>
          </div>

          {formError && (
            <div className="alert alert-danger" style={{ marginBottom: '16px', fontSize: '13px' }}>
              {formError}
            </div>
          )}

          <form onSubmit={handleCreate}>
            <div style={{ marginBottom: '16px' }}>
              <label
                htmlFor="kb-title"
                style={{ display: 'block', fontSize: '13px', fontWeight: 500, marginBottom: '6px' }}
              >
                Article Title <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>({title.length}/300)</span>
              </label>
              <input
                id="kb-title"
                type="text"
                className="form-input"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. Disputed Transaction Resolution Policy"
                maxLength={300}
                required
                disabled={submitting}
                style={{ width: '100%', padding: '8px 12px', fontSize: '13px' }}
              />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label
                htmlFor="kb-body"
                style={{ display: 'block', fontSize: '13px', fontWeight: 500, marginBottom: '6px' }}
              >
                Content Narrative{' '}
                <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>({body.length}/20,000)</span>
              </label>
              <textarea
                id="kb-body"
                className="form-input"
                value={body}
                onChange={(e) => setBody(e.target.value)}
                placeholder="Paste reference policy text, guidelines, or frequently asked dispute procedures..."
                rows={10}
                maxLength={20000}
                required
                disabled={submitting}
                style={{ width: '100%', padding: '10px 12px', fontSize: '13px', fontFamily: 'var(--font-sans)' }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowCreate(false)}
                disabled={submitting}
              >
                Cancel
              </button>
              <button type="submit" className="btn btn-primary" disabled={submitting}>
                {submitting ? 'Partitioning & Saving...' : 'Save & Partition Article'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Articles List */}
      <div className="card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-main)', margin: 0 }}>
            Published Articles ({articles.length})
          </h2>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={loadArticles}
            disabled={loading}
            style={{ fontSize: '12px', padding: '4px 10px' }}
          >
            {loading ? 'Refreshing...' : '↻ Refresh'}
          </button>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            Loading knowledge base articles...
          </div>
        ) : error ? (
          <div className="alert alert-danger" style={{ fontSize: '13px' }}>
            {error}
          </div>
        ) : articles.length === 0 ? (
          <div
            style={{
              backgroundColor: 'var(--bg-subtle)',
              border: '1px solid var(--border-color)',
              borderRadius: '4px',
              padding: '32px 16px',
              textAlign: 'center',
            }}
          >
            <p style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-main)', marginBottom: '4px' }}>
              No Knowledge Base Articles
            </p>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
              Add reference documentation to enable relevant passage retrieval on support complaints.
            </p>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setShowCreate(true)}
              style={{ fontSize: '13px', padding: '6px 14px' }}
            >
              + Add First Article
            </button>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600 }}>Article Title</th>
                  <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, width: '120px' }}>
                    Status
                  </th>
                  <th
                    style={{
                      padding: '8px 10px',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      textAlign: 'right',
                      width: '100px',
                    }}
                  >
                    Chunks
                  </th>
                  <th
                    style={{
                      padding: '8px 10px',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      textAlign: 'right',
                      width: '160px',
                    }}
                  >
                    Created
                  </th>
                </tr>
              </thead>
              <tbody>
                {articles.map((item) => (
                  <tr key={item.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '10px' }}>
                      <span style={{ fontWeight: 500, color: 'var(--text-main)' }}>{item.title}</span>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                        ID: <code className="text-mono">{item.id}</code>
                      </div>
                    </td>
                    <td style={{ padding: '10px' }}>
                      <span className={`badge badge-status-${item.status.toLowerCase()}`}>{item.status}</span>
                    </td>
                    <td
                      style={{
                        padding: '10px',
                        textAlign: 'right',
                        fontFamily: 'var(--font-mono)',
                        fontWeight: 500,
                      }}
                    >
                      {item.chunk_count}
                    </td>
                    <td style={{ padding: '10px', textAlign: 'right', color: 'var(--text-muted)', fontSize: '12px' }}>
                      {item.created_at ? new Date(item.created_at).toISOString().slice(0, 10) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
