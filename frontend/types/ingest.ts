// ── POST /tenant/ingest ───────────────────────────────────────────────────────

export interface IngestJobCreated {
  job_id: string
  status: "pending"
  message: string
}

// ── GET /tenant/ingest/{job_id} ───────────────────────────────────────────────

export type JobStatus = "pending" | "processing" | "completed" | "failed"

export interface IngestRow {
  tenant_sku_id: string
  product_id: string
  brand: string
  product_description: string
  package_type?: string
  delta_status: string
  action: "AUTO_MATCH" | "REVIEW_QUEUE" | "CREATE_NEW" | "UNCHANGED"
  matched_global_sku: string | null
  confidence: number
  match_signals: string[]
  reasoning: string
}

export interface IngestSummaryStats {
  total: number
  AUTO_MATCH: number
  REVIEW_QUEUE: number
  CREATE_NEW: number
  UNCHANGED: number
  alerts: number
}

export interface IngestResult {
  run_id: string
  source_file: string
  output_file: string
  started_at: string
  completed_at: string
  summary: IngestSummaryStats
  alerts: unknown[]
  rows: IngestRow[]
}

export interface IngestJobStatus {
  job_id: string
  status: JobStatus
  created_at: string
  started_at: string | null
  completed_at: string | null
  source_path: string | null
  output_path: string | null
  error: string | null
  result: IngestResult | null
}
