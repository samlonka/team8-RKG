// ── Request ──────────────────────────────────────────────────────────────────

export interface QueryRequest {
  question: string
  anchor_sku: string | null
}

// ── Agent pipeline models (mirrors agents/models.py) ─────────────────────────

export interface EntityNode {
  entity_id: string
  label: string
  display_name: string
  properties: Record<string, unknown>
  anomaly_score: number | null
  timestamp: string | null
  source: string
}

export interface ValidatedChain {
  chain_id: string
  path: EntityNode[]
  confidence: number
  temporal_validity: number
  evidence_density: number
  avg_anomaly_score: number
  reasoning: string
  source: string
}

export interface QueryTask {
  step: number
  task_type: string
  label: string
  description: string
  cypher: string | null
  cypher_params: Record<string, unknown>
  anchor_id: string | null
  index_name: string | null
  top_k: number
  use_self_emb: boolean
  use_reflect_emb: boolean
}

export interface QuerySpec {
  question: string
  task_type: string
  entity_types: string[]
  anchor_label: string | null
  anchor_entity_id: string | null
  time_window: { start: string; end: string } | null
  traversal_depth: number
}

// ── Response (mirrors api/main.py QueryResponse) ─────────────────────────────

export interface QueryResponse {
  question: string
  latency_seconds: number
  summary: string
  best_confidence: number | null
  best_classification: string | null
  best_reasoning: string | null
  planner_rationale: string
  validated_chains: ValidatedChain[]
  spec: QuerySpec | Record<string, unknown>
  tasks: QueryTask[]
}

// ── UI helpers ────────────────────────────────────────────────────────────────

export type ClassificationLabel = "Confirmed Anomaly" | "Needs Review" | "Healthy"

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  queryResponse?: QueryResponse
  isError?: boolean
}
