const API_BASE =
  process.env.NEXT_PUBLIC_RKG_API_URL ?? "http://127.0.0.1:8000"

export type PipelinePhase =
  | "supervisor"
  | "planner"
  | "doer"
  | "critic"
  | "complete"
  | "error"

export interface PipelineEvent {
  id: string
  phase: PipelinePhase
  status: "running" | "done" | "error"
  title: string
  detail?: string
  meta?: Record<string, unknown>
}

export interface EntityNodePayload {
  entity_id: string
  label: string
  display_name: string
  properties?: Record<string, unknown>
  anomaly_score?: number | null
  timestamp?: string | null
  source?: string
}

export interface ValidatedChainPayload {
  chain_id: string
  path: EntityNodePayload[]
  confidence: number
  temporal_validity: number
  evidence_density: number
  avg_anomaly_score: number
  reasoning: string
  source: string
}

export interface MatchedSkuRow {
  sku_id: string
  brand_name?: string
  package_category_name?: string
  package_name?: string
  status?: string
  confidence: number
  signals?: string[]
  score_breakdown?: Record<string, number>
}

export interface ProductRiskSummary {
  sku_id: string
  sku_anomaly?: number | null
  anomaly_max?: number | null
  anomaly_mean?: number | null
  anomaly_weighted?: number | null
  classification: string
  summary?: string
  drivers?: Array<{
    label: string
    entity_id: string
    display_name?: string
    anomaly: number
    relationship: string
    context?: string
  }>
}

export interface QueryAskResponse {
  question: string
  task_type: string
  summary: string
  scenario?: number | null
  best_confidence?: number
  best_classification?: string
  best_reasoning?: string
  doer_summary?: string
  best_chain?: ValidatedChainPayload
  validated_chains?: ValidatedChainPayload[]
  candidate_summaries?: Array<{ chain_id: string; summary: string }>
  latency_seconds?: number
  pipeline_events?: PipelineEvent[]
  catalog_query?: {
    brand_name: string
    package_type: string
    query_dims?: Record<string, number>
  }
  product_risk?: ProductRiskSummary
  match_result?: {
    status: string
    confidence: number
    reasoning?: string
    ambiguous?: boolean
    dim_applied?: boolean
    product_risk?: ProductRiskSummary
    matched_skus?: MatchedSkuRow[]
    pipeline?: Record<string, number>
  }
  duplicate_report?: {
    has_duplicates: boolean
    total_groups: number
    total_groups_shown?: number
    upc_groups_total?: number
    brand_package_groups_total?: number
    source: string
    upc_duplicate_groups?: Array<{ upc: string; sku_ids: string[]; count: number }>
    brand_package_duplicate_groups?: Array<{
      brand_name: string
      package_type: string
      sku_ids: string[]
      count: number
    }>
  }
  closed_world_rows?: Array<Record<string, unknown>> | null
  reflexive_finding?: string
  display_limits?: Array<{
    key: string
    label: string
    shown: number
    total: number
  }>
}

function parseSseChunk(chunk: string): PipelineEvent | { phase: "complete"; result: QueryAskResponse } | { phase: "error"; detail: string } | null {
  const line = chunk
    .split("\n")
    .find((l) => l.startsWith("data: "))
  if (!line) return null
  try {
    return JSON.parse(line.slice(6))
  } catch {
    return null
  }
}

export async function queryAskStream(
  question: string,
  onEvent: (event: PipelineEvent) => void
): Promise<QueryAskResponse> {
  const res = await fetch(`${API_BASE}/query/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  })

  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `API error ${res.status}`)
  }

  const reader = res.body?.getReader()
  if (!reader) {
    throw new Error("Streaming not supported by the browser")
  }

  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split("\n\n")
    buffer = parts.pop() ?? ""

    for (const part of parts) {
      const parsed = parseSseChunk(part)
      if (!parsed) continue

      if ("result" in parsed && parsed.phase === "complete") {
        return parsed.result as QueryAskResponse
      }
      if ("detail" in parsed && parsed.phase === "error" && !("title" in parsed)) {
        throw new Error(String(parsed.detail))
      }
      if ("phase" in parsed && parsed.phase !== "complete") {
        onEvent(parsed as PipelineEvent)
        if (parsed.phase === "error" && "detail" in parsed) {
          throw new Error(String((parsed as PipelineEvent).detail || "Pipeline failed"))
        }
      }
    }
  }

  throw new Error("Stream ended without a final result")
}

export async function queryAsk(question: string): Promise<QueryAskResponse> {
  const res = await fetch(`${API_BASE}/query/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `API error ${res.status}`)
  }
  return res.json()
}

export function formatQueryResponse(data: QueryAskResponse): string {
  const lines: string[] = []

  if (data.best_classification) {
    lines.push(data.best_classification)
  }

  if (data.task_type === "catalog_match") {
    const top = data.match_result?.matched_skus?.[0]
    const status = data.match_result?.status
    const conf = data.best_confidence ?? data.match_result?.confidence
    const brand = data.catalog_query?.brand_name
    const pkg = data.catalog_query?.package_type

    if (top && status && status !== "insert") {
      lines.push(
        `Best match: GlobalSKU ${top.sku_id}${conf != null ? ` · ${(conf * 100).toFixed(0)}% confidence` : ""}.`
      )
    } else if (data.match_result?.reasoning) {
      lines.push(data.match_result.reasoning.split(". ")[0] + ".")
    } else {
      lines.push("No strong match in the master catalog for this brand and package.")
    }
    if (brand || pkg) {
      lines.push(`Query: ${brand ?? "—"} · ${pkg ?? "—"}`)
    }
  } else if (data.task_type === "catalog_duplicate") {
    const dr = data.duplicate_report
    if (dr?.has_duplicates) {
      const src = dr.source ? ` (${dr.source})` : ""
      lines.push(
        `Yes — ${dr.total_groups} duplicate group(s) in the master catalog${src}.`
      )
    } else {
      lines.push("No duplicate UPC or brand+package groups in the master catalog.")
    }
  } else if (!data.doer_summary && data.best_reasoning) {
    const first = data.best_reasoning.split(". ").slice(0, 2).join(". ")
    if (first) lines.push(first + (first.endsWith(".") ? "" : "."))
  }

  if (data.best_confidence != null) {
    lines.push(`Confidence: ${(data.best_confidence * 100).toFixed(0)}%`)
  }

  for (const limit of data.display_limits ?? []) {
    if (limit.total > limit.shown) {
      lines.push(`Showing top ${limit.shown} of ${limit.total} ${limit.label}.`)
    }
  }

  return lines.join("\n")
}
