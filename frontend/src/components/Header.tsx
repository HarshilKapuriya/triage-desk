import { User } from '../types';

interface HeaderProps {
  user: User | null;
  onLogout: () => void;
  onNavigateHome: () => void;
}

export const Header = ({ user, onLogout, onNavigateHome }: HeaderProps) => {
  return (
    <header className="top-nav">
      <button type="button" className="brand" onClick={onNavigateHome} aria-label="Triage Desk Home">
        <div className="brand-mark">TD</div>
        <div className="brand-title">Triage Desk</div>
        <span className="brand-tag">React · Read-Only</span>
      </button>

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

