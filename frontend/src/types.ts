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
  first_response_at?: string | null;
  resolved_at?: string | null;
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

export interface KBArticleSummary {
  id: string;
  title: string;
  status: 'processing' | 'ready' | 'failed' | string;
  chunk_count: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface RelevantKnowledgeItem {
  chunk_id: string;
  article_id: string;
  article_title: string;
  chunk_index: number;
  excerpt: string;
  char_count: number;
  token_count: number;
  similarity_score: number;
}

export interface RelevantKnowledgeResponse {
  ticket_id: string;
  status: 'ready' | 'missing_ticket_embedding' | 'articles_processing' | 'no_chunks_available' | string;
  model_name: string | null;
  model_version: string | null;
  items: RelevantKnowledgeItem[];
  notice: string;
  message?: string | null;
}

export interface SLATargetEvaluation {
  target_name: string;
  target_minutes: number;
  due_at: string;
  is_completed: boolean;
  completed_at: string | null;
  is_breached: boolean;
  status: string; // "fulfilled", "breached", "active"
  timeliness: string; // "on_time", "late", "on_track", "overdue"
  timeliness_label?: string; // "Due now", "Overdue", "On track", "Fulfilled (on time)", "Breached (late)"
  elapsed_minutes: number;
  elapsed_fraction: number;
  urgency: string; // "low", "medium", "high", "overdue", "completed"
}

export interface TicketSLAResponse {
  schema_available: boolean;
  is_configured: boolean;
  policy_priority: string | null;
  overall_status: string; // "schema_unavailable", "not_configured", "active", "breached", "fulfilled"
  overall_urgency: string | null; // "low", "medium", "high", "overdue", "completed", null
  evaluated_at: string;
  first_response: SLATargetEvaluation | null;
  resolution: SLATargetEvaluation | null;
  urgency_notice: string;
  message?: string | null;
}

export interface LifecycleTicketInfo {
  id: string;
  status: string;
  first_response_at: string | null;
  resolved_at: string | null;
}

export interface LifecycleActionResponse {
  ok: boolean;
  ticket: LifecycleTicketInfo;
}

export interface SLAPolicyItem {
  id: string;
  priority: string;
  first_response_minutes: number;
  resolution_minutes: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface SLAPoliciesResponse {
  schema_available: boolean;
  policies: SLAPolicyItem[];
  message?: string;
}

export interface SLAPolicyMutationResponse {
  schema_available: boolean;
  policy: SLAPolicyItem;
}
