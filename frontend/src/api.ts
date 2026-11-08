import { User, TicketListItem, Pagination, TicketDetailResponse, SimilarTicketsResponse, SimilarFeedbackResponse } from './types';

class ApiClient {
  private csrfToken: string | null = null;

  private async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const headers = new Headers(options.headers || {});
    headers.set('Accept', 'application/json');

    // Attach CSRF token for mutating requests if available
    const method = (options.method || 'GET').toUpperCase();
    if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
      if (!this.csrfToken) {
        await this.fetchCsrfToken();
      }
      if (this.csrfToken) {
        headers.set('X-CSRF-Token', this.csrfToken);
      }
    }

    const response = await fetch(path, {
      ...options,
      headers,
      credentials: 'same-origin',
    });

    if (!response.ok) {
      let errorMessage = `Request failed with status ${response.status}`;
      try {
        const errorJson = await response.json();
        if (errorJson?.error?.message) {
          errorMessage = errorJson.error.message;
        }
      } catch {
        // Fall back to HTTP status message
      }
      throw new Error(errorMessage);
    }

    return response.json() as Promise<T>;
  }

  public async fetchCsrfToken(): Promise<string> {
    const data = await this.request<{ csrf_token: string }>('/browser-api/csrf-token');
    this.csrfToken = data.csrf_token;
    return this.csrfToken;
  }

  public async getSession(): Promise<{ authenticated: boolean; user: User } | null> {
    try {
      const data = await this.request<{ authenticated: boolean; user: User; csrf_token: string }>(
        '/browser-api/auth/session'
      );
      if (data.csrf_token) {
        this.csrfToken = data.csrf_token;
      }
      return data;
    } catch {
      return null;
    }
  }

  public async login(email: string, password: string): Promise<User> {
    await this.fetchCsrfToken();
    const data = await this.request<{ user: User; csrf_token: string }>('/browser-api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, csrf_token: this.csrfToken }),
    });
    this.csrfToken = data.csrf_token;
    return data.user;
  }

  public async logout(): Promise<void> {
    try {
      await this.request<{ status: string }>('/browser-api/auth/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ csrf_token: this.csrfToken }),
      });
    } finally {
      this.csrfToken = null;
    }
  }

  public async getTickets(page = 1, perPage = 20): Promise<{ tickets: TicketListItem[]; pagination: Pagination }> {
    return this.request<{ tickets: TicketListItem[]; pagination: Pagination }>(
      `/browser-api/tickets?page=${page}&per_page=${perPage}`
    );
  }

  public async getTicketDetail(ticketId: string): Promise<TicketDetailResponse> {
    return this.request<TicketDetailResponse>(`/browser-api/tickets/${ticketId}`);
  }

  public async getSimilarTickets(ticketId: string, limit = 5): Promise<SimilarTicketsResponse> {
    return this.request<SimilarTicketsResponse>(`/browser-api/tickets/${ticketId}/similar?limit=${limit}`);
  }

  public async createTicket(
    subject: string,
    body: string
  ): Promise<{ id: string; ticket: { id: string; status: string; subject: string; created_at: string | null } }> {
    return this.request<{ id: string; ticket: { id: string; status: string; subject: string; created_at: string | null } }>(
      '/browser-api/tickets',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject, body, csrf_token: this.csrfToken }),
      }
    );
  }

  public async submitSimilarFeedback(
    ticketId: string,
    similarTicketId: string,
    decision: 'confirmed' | 'rejected'
  ): Promise<SimilarFeedbackResponse> {
    if (!this.csrfToken) {
      await this.fetchCsrfToken();
    }
    return this.request<SimilarFeedbackResponse>(
      `/browser-api/tickets/${ticketId}/similar/${similarTicketId}/feedback`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, csrf_token: this.csrfToken }),
      }
    );
  }
}

export const api = new ApiClient();
