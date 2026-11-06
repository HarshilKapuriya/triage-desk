import { useState } from 'react';
import { api } from '../api';
import { User } from '../types';

interface LoginViewProps {
  onLoginSuccess: (user: User) => void;
}

export const LoginView = ({ onLoginSuccess }: LoginViewProps) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const user = await api.login(email, password);
      onLoginSuccess(user);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Invalid email or password.');
    } finally {
      setLoading(false);
      // Clear password field on submission attempt
      setPassword('');
    }
  };

  return (
    <div style={{ maxWidth: '420px', margin: '40px auto' }}>
      <div className="card" style={{ padding: '28px 24px' }}>
        <div style={{ marginBottom: '24px', textAlign: 'left' }}>
          <h1 style={{ fontSize: '20px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '6px' }}>
            Sign In
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            Access the read-only triage desk to review incoming financial tickets and automated product classifications.
          </p>
        </div>

        {error && (
          <div className="alert alert-danger" role="alert">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} noValidate>
          <div className="form-group">
            <label className="form-label" htmlFor="login-email">
              Work Email
            </label>
            <input
              id="login-email"
              className="form-input"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="username"
              autoFocus
              placeholder="agent@company.com"
              disabled={loading}
            />
          </div>

          <div className="form-group" style={{ marginBottom: '24px' }}>
            <label className="form-label" htmlFor="login-password">
              Password
            </label>
            <input
              id="login-password"
              className="form-input"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
              placeholder="••••••••"
              disabled={loading}
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            style={{ width: '100%', padding: '10px 16px' }}
            disabled={loading}
          >
            {loading ? 'Signing In...' : 'Sign In to Triage Desk'}
          </button>
        </form>
      </div>
    </div>
  );
};

