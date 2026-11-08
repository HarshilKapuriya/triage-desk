export interface User {
  id: string;
  tenant_id: string;
  role: string;
  email?: string | null;
}

export interface ProcessingStatus {
  state: 'queued' | 'processing' | 'retry_scheduled' | 'completed' | 'failed' | 'not_found' | string;
  label: string;
  details?: string | null;
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
  processing_status: ProcessingStatus;
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
  processing: ProcessingStatus;
  notices: DomainNotices;
}

export interface SimilarTicketItem {
  ticket_id: string;
  external_id: string | null;
  subject: string;
  status: string;
  priority: string | null;
  created_at: string | null;
  similarity_score: number;
  decision_status?: 'suggested' | 'confirmed' | 'rejected' | null;
  reviewed_at?: string | null;
}

export interface SimilarFeedbackResponse {
  link_id: string;
  source_ticket_id: string;
  target_ticket_id: string;
  similarity_score: number;
  model_name: string;
  model_version: string;
  decision: 'confirmed' | 'rejected';
  reviewed_at: string | null;
  notice: string;
}

export interface SimilarTicketsResponse {
  ticket_id: string;
  has_embedding: boolean;
  embedding_status: 'ready' | 'not_ready' | string;
  model_name: string | null;
  model_version: string | null;
  items: SimilarTicketItem[];
  notice: string;
  message?: string | null;
}
