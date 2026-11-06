export interface User {
  id: string;
  tenant_id: string;
  role: string;
  email?: string | null;
}

export interface TicketListItem {
  id: string;
  external_id: string | null;
  subject: string;
  customer_email: string | null;
  status: string;
  category: string | null;
  priority: string | null;
  created_at: string | null;
  predicted_label: string | null;
  model_name: string | null;
  model_version: string | null;
}

export interface Pagination {
  page: number;
  per_page: number;
  total_count: number;
  total_pages: number;
}

export interface TicketDetail {
  id: string;
  external_id: string | null;
  subject: string;
  body: string;
  customer_email: string | null;
  status: string;
  category: string | null;
  priority: string | null;
  created_at: string | null;
}

export interface ModelPrediction {
  predicted_label: string;
  model_name: string;
  model_version: string;
  confidence: number | null;
  confidence_display: string;
  decision_scores: Record<string, number> | null;
  created_at: string | null;
}

export interface DomainNotices {
  domain_notice: string;
  confidence_notice: string;
  priority_notice: string;
}

export interface TicketDetailResponse {
  ticket: TicketDetail;
  prediction: ModelPrediction | null;
  notices: DomainNotices;
}

