import { User } from '../types';

interface HeaderProps {
  user: User | null;
  onLogout: () => void;
  onNavigateHome: () => void;
  currentView?: string;
  onNavigateKB?: () => void;
  onNavigateDashboard?: () => void;
  onOpenSlaModal?: () => void;
}

export const Header = ({
  user,
  onLogout,
  onNavigateHome,
  currentView,
  onNavigateKB,
  onNavigateDashboard,
  onOpenSlaModal,
}: HeaderProps) => {
  return (
    <header className="top-nav">
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        <button type="button" className="brand" onClick={onNavigateHome} aria-label="Triage Desk Home">
          <div className="brand-mark">TD</div>
          <div className="brand-title">Triage Desk</div>
          <span className="brand-tag">React · Read-Only</span>
        </button>

        {user && onNavigateKB && (
          <nav className="nav-segmented-dock">
            {onNavigateDashboard && (
              <button
                type="button"
                onClick={onNavigateDashboard}
                className={`nav-pill-btn ${currentView === 'dashboard' ? 'active' : ''}`}
              >
                Dashboard
              </button>
            )}
            <button
              type="button"
              onClick={onNavigateHome}
              className={`nav-pill-btn ${currentView === 'tickets' ? 'active' : ''}`}
            >
              Tickets Queue
            </button>
            <button
              type="button"
              onClick={onNavigateKB}
              className={`nav-pill-btn ${currentView === 'knowledge-base' ? 'active' : ''}`}
            >
              Knowledge Base
            </button>
          </nav>
        )}
      </div>

      <div className="nav-actions">
        {user && (
          <>
            {user.role === 'admin' && onOpenSlaModal && (
              <button
                type="button"
                onClick={onOpenSlaModal}
                className="btn-header-action"
                title="Configure Tenant SLA Policies (Admin only)"
              >
                ⚙ SLA Policies
              </button>
            )}
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

