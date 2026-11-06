import { useEffect, useState } from 'react';
import { api } from './api';
import { User, TicketListItem, Pagination } from './types';
import { NoticeBanner } from './components/NoticeBanner';
import { Header } from './components/Header';
import { LoginView } from './components/LoginView';
import { TicketsListView } from './components/TicketsListView';
import { TicketDetailView } from './components/TicketDetailView';
import { CreateTicketView } from './components/CreateTicketView';
import './App.css';

type ViewMode = 'login' | 'tickets' | 'ticket-detail' | 'new-ticket';

export const App = () => {
  const [user, setUser] = useState<User | null>(null);
  const [view, setView] = useState<ViewMode>('login');
  const [selectedTicketId, setSelectedTicketId] = useState<string | null>(null);
  const [tickets, setTickets] = useState<TicketListItem[]>([]);
  const [pagination, setPagination] = useState<Pagination>({
    page: 1,
    per_page: 20,
    total_count: 0,
    total_pages: 1,
  });
  const [loading, setLoading] = useState(true);

  // Sync initial URL path
  const syncFromPath = (currentUser: User | null) => {
    if (!currentUser) {
      setView('login');
      setSelectedTicketId(null);
      return;
    }
    const path = window.location.pathname;
    if (path === '/tickets/new') {
      setSelectedTicketId(null);
      setView('new-ticket');
      return;
    }
    const match = path.match(/^\/tickets\/([a-zA-Z0-9-]+)$/);
    if (match) {
      setSelectedTicketId(match[1]);
      setView('ticket-detail');
    } else {
      setSelectedTicketId(null);
      setView('tickets');
    }
  };

  // Check existing session on load
  useEffect(() => {
    let isMounted = true;
    api
      .getSession()
      .then((session) => {
        if (!isMounted) return;
        if (session && session.authenticated) {
          setUser(session.user);
          syncFromPath(session.user);
        } else {
          setUser(null);
          setView('login');
        }
      })
      .catch(() => {
        if (isMounted) {
          setUser(null);
          setView('login');
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    const handlePopState = () => {
      syncFromPath(user);
    };
    window.addEventListener('popstate', handlePopState);

    return () => {
      isMounted = false;
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  // Fetch tickets whenever on the tickets view or page changes
  const fetchTickets = (page = 1) => {
    setLoading(true);
    api
      .getTickets(page, 20)
      .then((res) => {
        setTickets(res.tickets);
        setPagination(res.pagination);
      })
      .catch((err: unknown) => {
        // If unauthenticated, redirect to login
        if (err instanceof Error && err.message.toLowerCase().includes('authentication')) {
          setUser(null);
          setView('login');
        }
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    if (user && view === 'tickets') {
      fetchTickets(pagination.page);
    }
  }, [user, view, pagination.page]);

  const handleLoginSuccess = (loggedInUser: User) => {
    setUser(loggedInUser);
    setView('tickets');
    setSelectedTicketId(null);
    window.history.pushState({}, '', '/tickets');
    fetchTickets(1);
  };

  const handleLogout = async () => {
    try {
      await api.logout();
    } finally {
      setUser(null);
      setView('login');
      setSelectedTicketId(null);
      setTickets([]);
      window.history.pushState({}, '', '/login');
    }
  };

  const handleSelectTicket = (id: string) => {
    setSelectedTicketId(id);
    setView('ticket-detail');
    window.history.pushState({}, '', `/tickets/${id}`);
  };

  const handleNewTicket = () => {
    setSelectedTicketId(null);
    setView('new-ticket');
    window.history.pushState({}, '', '/tickets/new');
  };

  const handleTicketCreated = (id: string) => {
    setSelectedTicketId(id);
    setView('ticket-detail');
    window.history.pushState({}, '', `/tickets/${id}`);
  };

  const handleBackToQueue = () => {
    setSelectedTicketId(null);
    setView('tickets');
    window.history.pushState({}, '', '/tickets');
  };

  const handlePageChange = (newPage: number) => {
    setPagination((prev) => ({ ...prev, page: newPage }));
  };

  return (
    <div className="app-wrapper">
      <NoticeBanner />
      <Header
        user={user}
        onLogout={handleLogout}
        onNavigateHome={() => {
          if (user) handleBackToQueue();
        }}
      />

      <main className="main-content">
        {loading && !user && (
          <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)' }}>
            Initializing Triage Desk...
          </div>
        )}

        {!loading && !user && <LoginView onLoginSuccess={handleLoginSuccess} />}

        {user && view === 'tickets' && (
          <TicketsListView
            tickets={tickets}
            pagination={pagination}
            loading={loading}
            onPageChange={handlePageChange}
            onSelectTicket={handleSelectTicket}
            onNewTicket={handleNewTicket}
          />
        )}

        {user && view === 'new-ticket' && (
          <CreateTicketView
            onSuccess={handleTicketCreated}
            onCancel={handleBackToQueue}
          />
        )}

        {user && view === 'ticket-detail' && selectedTicketId && (
          <TicketDetailView ticketId={selectedTicketId} onBack={handleBackToQueue} />
        )}
      </main>

      <footer className="app-footer">
        Triage Desk Prototype · CFPB Financial Product Classifier (LinearSVC balanced run_004) · React + Vite + TS
        Interface
      </footer>
    </div>
  );
};

