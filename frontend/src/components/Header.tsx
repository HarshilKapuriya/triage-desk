import { User } from '../types';

interface HeaderProps {
  user: User | null;
  onLogout: () => void;
  onNavigateHome: () => void;
  currentView?: string;
  onNavigateKB?: () => void;
}

export const Header = ({ user, onLogout, onNavigateHome, currentView, onNavigateKB }: HeaderProps) => {
  return (
    <header className="top-nav">
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        <button type="button" className="brand" onClick={onNavigateHome} aria-label="Triage Desk Home">
          <div className="brand-mark">TD</div>
          <div className="brand-title">Triage Desk</div>
          <span className="brand-tag">React · Read-Only</span>
        </button>

        {user && onNavigateKB && (
          <nav style={{ display: 'flex', gap: '8px' }}>
            <button
              type="button"
              onClick={onNavigateHome}
              style={{
                background: currentView === 'tickets' ? 'var(--bg-subtle)' : 'none',
                border: '1px solid',
                borderColor: currentView === 'tickets' ? 'var(--border-color)' : 'transparent',
                borderRadius: '4px',
                padding: '4px 10px',
                fontSize: '13px',
                fontWeight: currentView === 'tickets' ? 600 : 400,
                color: 'var(--text-main)',
                cursor: 'pointer',
              }}
            >
              Tickets Queue
            </button>
            <button
              type="button"
              onClick={onNavigateKB}
              style={{
                background: currentView === 'knowledge-base' ? 'var(--bg-subtle)' : 'none',
                border: '1px solid',
                borderColor: currentView === 'knowledge-base' ? 'var(--border-color)' : 'transparent',
                borderRadius: '4px',
                padding: '4px 10px',
                fontSize: '13px',
                fontWeight: currentView === 'knowledge-base' ? 600 : 400,
                color: 'var(--text-main)',
                cursor: 'pointer',
              }}
            >
              Knowledge Base
            </button>
          </nav>
        )}
      </div>

      <div className="nav-actions">
        {user && (
          <>
            <span className="nav-user">
              Tenant: <code title={user.tenant_id}>{user.tenant_id.slice(0, 8)}...</code> · Role:{' '}
              <strong>{user.role}</strong>
            </span>
            <button type="button" className="btn-logout" onClick={onLogout}>
              Sign Out
            </button>
          </>
        )}
      </div>
    </header>
  );
};

