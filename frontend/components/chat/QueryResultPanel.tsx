"use client"

import type { ReactNode } from "react"
import type { QueryAskResponse } from "@/lib/rkg-api"
import { DuplicateReportCollapsible } from "@/components/chat/PipelineProgress"

type EntityNode = {
  entity_id: string
  label: string
  display_name: string
  properties?: Record<string, unknown>
  anomaly_score?: number | null
  timestamp?: string | null
  source?: string
}

type ValidatedChain = {
  chain_id: string
  path: EntityNode[]
  confidence: number
  temporal_validity: number
  evidence_density: number
  avg_anomaly_score: number
  reasoning: string
  source: string
}

function pct(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—"
  return `${(n * 100).toFixed(0)}%`
}

function score(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—"
  return n.toFixed(2)
}

function TableShell({
  title,
  children,
}: {
  title: string
  children: ReactNode
}) {
  return (
    <section className="mt-3 border-t border-border/60 pt-3">
      <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <div className="overflow-x-auto rounded-md border border-border/60">
        {children}
      </div>
    </section>
  )
}

function Th({ children }: { children: ReactNode }) {
  return (
    <th className="px-2 py-1.5 text-left font-medium text-foreground/80">
      {children}
    </th>
  )
}

function Td({ children, mono }: { children: ReactNode; mono?: boolean }) {
  return (
    <td
      className={`border-t border-border/40 px-2 py-1.5 align-top text-muted-foreground ${
        mono ? "font-mono text-[10px]" : ""
      }`}
    >
      {children}
    </td>
  )
}

function evidenceTableTitle(
  chainId: string | undefined,
  path: EntityNode[],
  limits?: QueryAskResponse["display_limits"]
): string {
  const limitFor = (key: string) => limits?.find((l) => l.key === key)
  const suffix = (key: string, base: string) => {
    const l = limitFor(key)
    return l && l.total > l.shown ? `${base} (top ${l.shown} of ${l.total})` : base
  }
  if (chainId === "dynamic_rank") {
    return suffix("dynamic_rank", "Ranked GlobalSKUs by anomaly score")
  }
  if (chainId?.startsWith("anchor_")) {
    return `Evidence chain · GlobalSKU ${chainId.replace("anchor_", "")}`
  }
  if (chainId === "scenario3_top20") {
    return suffix("scenario3_risk_rank", "Ranked GlobalSKUs")
  }
  if (chainId === "scenario5") {
    return suffix("scenario5_shared_sku", "Shared GlobalSKUs (cross-customer risk)")
  }
  if (chainId === "scenario6") {
    return suffix("scenario6_auto_map", "Wrong auto-map SKUs")
  }
  if (chainId === "catalog_duplicate") return "Duplicate groups (evidence chain)"
  if (
    path.length > 1 &&
    path.every((n) => n.label === "GlobalSKU") &&
    path.some((n) => n.anomaly_score != null)
  ) {
    return "Ranked GlobalSKUs"
  }
  return "Evidence path"
}

function EvidencePathTable({
  path,
  chainId,
  displayLimits,
}: {
  path: EntityNode[]
  chainId?: string
  displayLimits?: QueryAskResponse["display_limits"]
}) {
  if (path.length === 0) return null

  return (
    <TableShell title={evidenceTableTitle(chainId, path, displayLimits)}>
      <table className="w-full min-w-[28rem] text-[11px]">
        <thead className="bg-background/60">
          <tr>
            <Th>#</Th>
            <Th>Type</Th>
            <Th>Entity</Th>
            <Th>Detail</Th>
            <Th>Anomaly</Th>
            <Th>Time</Th>
          </tr>
        </thead>
        <tbody>
          {path.map((node, i) => (
            <tr key={`${node.label}-${node.entity_id}-${i}`}>
              <Td mono>{String(i + 1)}</Td>
              <Td>{node.label}</Td>
              <Td mono>{node.entity_id}</Td>
              <Td>{node.display_name}</Td>
              <Td mono>{score(node.anomaly_score)}</Td>
              <Td mono>{node.timestamp || "—"}</Td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableShell>
  )
}

function MatchedSkusTable({
  data,
  displayLimits,
}: {
  data: NonNullable<QueryAskResponse["match_result"]>
  displayLimits?: QueryAskResponse["display_limits"]
}) {
  const rows = data.matched_skus ?? []
  if (rows.length === 0) return null

  const limit = displayLimits?.find((l) => l.key === "catalog_match")
  const title = limit && limit.total > limit.shown
    ? `Master catalog matches (${data.status}) · top ${limit.shown} of ${limit.total}`
    : `Master catalog matches (${data.status})`

  return (
    <TableShell title={title}>
      <table className="w-full min-w-[32rem] text-[11px]">
        <thead className="bg-background/60">
          <tr>
            <Th>SKU</Th>
            <Th>Brand</Th>
            <Th>Package</Th>
            <Th>Confidence</Th>
            <Th>Status</Th>
            <Th>Signals</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.sku_id}>
              <Td mono>{row.sku_id}</Td>
              <Td>{row.brand_name || "—"}</Td>
              <Td>{row.package_category_name || row.package_name || "—"}</Td>
              <Td mono>{pct(row.confidence)}</Td>
              <Td>{row.status || "—"}</Td>
              <Td mono>
                {row.signals?.length ? row.signals.join(", ") : "—"}
              </Td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableShell>
  )
}

function ProductRiskTable({
  risk,
}: {
  risk: NonNullable<QueryAskResponse["product_risk"]>
}) {
  const drivers = risk.drivers ?? []
  return (
    <>
      {risk.summary ? (
        <p className="mt-3 border-t border-border/60 pt-3 text-xs leading-relaxed text-muted-foreground">
          {risk.summary}
        </p>
      ) : null}
      {drivers.length > 0 ? (
        <TableShell title="Product risk drivers">
          <table className="w-full min-w-[28rem] text-[11px]">
            <thead className="bg-background/60">
              <tr>
                <Th>Type</Th>
                <Th>Entity</Th>
                <Th>Relationship</Th>
                <Th>Anomaly</Th>
                <Th>Context</Th>
              </tr>
            </thead>
            <tbody>
              {drivers.map((d) => (
                <tr key={`${d.label}-${d.entity_id}`}>
                  <Td>{d.label}</Td>
                  <Td mono>{d.entity_id}</Td>
                  <Td>{d.relationship}</Td>
                  <Td mono>{score(d.anomaly)}</Td>
                  <Td>{d.context || d.display_name || "—"}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableShell>
      ) : null}
    </>
  )
}

function ClosedWorldTable({ rows }: { rows: Record<string, unknown>[] }) {
  return (
    <TableShell title="Closed-world query (rule-based)">
      {rows.length === 0 ? (
        <p className="px-3 py-2 text-xs text-muted-foreground">
          0 rows — no <code className="font-mono">flag=&apos;duplicate&apos;</code>{" "}
          brands found (expected blind spot).
        </p>
      ) : (
        <table className="w-full text-[11px]">
          <thead className="bg-background/60">
            <tr>
              <Th>SKU</Th>
              <Th>Brand</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={String(row.sku_id ?? i)}>
                <Td mono>{String(row.sku_id ?? "—")}</Td>
                <Td>{String(row.brand ?? row.brand_family ?? "—")}</Td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </TableShell>
  )
}

function MetricsRow({ chain }: { chain: ValidatedChain }) {
  return (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-muted-foreground">
      <span>Temporal {pct(chain.temporal_validity)}</span>
      <span>Density {pct(chain.evidence_density)}</span>
      <span>Anomaly {score(chain.avg_anomaly_score)}</span>
    </div>
  )
}

function DisplayLimitsNote({
  limits,
}: {
  limits: NonNullable<QueryAskResponse["display_limits"]>
}) {
  const truncated = limits.filter((l) => l.total > l.shown)
  if (truncated.length === 0) return null

  return (
    <p className="mt-2 text-[10px] italic text-muted-foreground">
      {truncated
        .map((l) => `Showing top ${l.shown} of ${l.total} ${l.label}.`)
        .join(" ")}
    </p>
  )
}

interface QueryResultPanelProps {
  data: QueryAskResponse
}

/** Renders structured tables and LLM narratives for every pipeline response type. */
export function QueryResultPanel({ data }: QueryResultPanelProps) {
  const bestChain = (data.validated_chains?.[0] ?? data.best_chain) as
    | ValidatedChain
    | undefined
  const productRisk =
    data.product_risk ?? data.match_result?.product_risk ?? undefined

  return (
    <div className="mt-1 space-y-1">
      {data.display_limits && data.display_limits.length > 0 ? (
        <DisplayLimitsNote limits={data.display_limits} />
      ) : null}

      {data.doer_summary ? (
        <section className="mt-3 border-t border-border/60 pt-3">
          <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            Doer analysis
          </p>
          <p className="text-xs leading-relaxed text-foreground/90">
            {data.doer_summary}
          </p>
        </section>
      ) : null}

      {data.task_type === "catalog_match" && data.match_result ? (
        <MatchedSkusTable
          data={data.match_result}
          displayLimits={data.display_limits}
        />
      ) : null}

      {data.task_type === "catalog_duplicate" && data.duplicate_report ? (
        <DuplicateReportCollapsible
          report={data.duplicate_report}
          defaultOpen
        />
      ) : null}

      {data.scenario === 4 && data.closed_world_rows != null ? (
        <ClosedWorldTable rows={data.closed_world_rows} />
      ) : null}

      {bestChain?.path?.length ? (
        <>
          <EvidencePathTable
            path={bestChain.path}
            chainId={bestChain.chain_id}
            displayLimits={data.display_limits}
          />
          <MetricsRow chain={bestChain} />
        </>
      ) : null}

      {data.validated_chains && data.validated_chains.length > 1 ? (
        <TableShell title={`Additional validated chains (${data.validated_chains.length - 1})`}>
          <table className="w-full text-[11px]">
            <thead className="bg-background/60">
              <tr>
                <Th>Chain</Th>
                <Th>Entities</Th>
                <Th>Confidence</Th>
                <Th>Anomaly</Th>
              </tr>
            </thead>
            <tbody>
              {data.validated_chains.slice(1).map((chain) => (
                <tr key={chain.chain_id}>
                  <Td mono>{chain.chain_id}</Td>
                  <Td mono>{chain.path?.length ?? 0}</Td>
                  <Td mono>{pct(chain.confidence)}</Td>
                  <Td mono>{score(chain.avg_anomaly_score)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableShell>
      ) : null}

      {productRisk ? <ProductRiskTable risk={productRisk} /> : null}

      {data.best_reasoning ? (
        <section className="mt-3 border-t border-border/60 pt-3">
          <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            Critic validation
          </p>
          <p className="text-xs leading-relaxed text-muted-foreground">
            {data.best_reasoning}
          </p>
        </section>
      ) : null}

      {data.match_result?.reasoning &&
      data.task_type === "catalog_match" ? (
        <section className="mt-3 border-t border-border/60 pt-3">
          <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            Match reasoning
          </p>
          <p className="text-xs leading-relaxed text-muted-foreground">
            {data.match_result.reasoning}
          </p>
        </section>
      ) : null}

      {data.scenario === 4 && data.reflexive_finding ? (
        <p className="mt-2 text-xs text-foreground/80">{data.reflexive_finding}</p>
      ) : null}
    </div>
  )
}
