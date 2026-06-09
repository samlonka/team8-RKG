"use client"

import { Download, CheckCircle2, AlertTriangle, RefreshCw, PlusCircle, Minus } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import { getDownloadUrl } from "@/services/ingestApi"
import type { IngestJobStatus, IngestRow } from "@/types/ingest"

// ── Action badge ──────────────────────────────────────────────────────────────

const ACTION_CONFIG = {
  AUTO_MATCH: {
    label: "Auto Match",
    className: "bg-emerald-500/10 text-emerald-700 border-emerald-500/20",
    icon: <CheckCircle2 className="size-3" />,
  },
  REVIEW_QUEUE: {
    label: "Review Queue",
    className: "bg-amber-500/10 text-amber-700 border-amber-500/20",
    icon: <AlertTriangle className="size-3" />,
  },
  CREATE_NEW: {
    label: "Create New",
    className: "bg-blue-500/10 text-blue-700 border-blue-500/20",
    icon: <PlusCircle className="size-3" />,
  },
  UNCHANGED: {
    label: "Unchanged",
    className: "bg-muted text-muted-foreground border-border",
    icon: <Minus className="size-3" />,
  },
} as const

function ActionBadge({ action }: { action: IngestRow["action"] }) {
  const cfg = ACTION_CONFIG[action] ?? ACTION_CONFIG.UNCHANGED
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium",
        cfg.className
      )}
    >
      {cfg.icon}
      {cfg.label}
    </span>
  )
}

// ── Confidence bar ────────────────────────────────────────────────────────────

function ConfidencePill({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 90 ? "text-emerald-600" : pct >= 70 ? "text-amber-600" : "text-destructive"
  return (
    <span className={cn("text-xs font-medium tabular-nums", color)}>{pct}%</span>
  )
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  highlight,
}: {
  label: string
  value: number
  highlight?: boolean
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center rounded-lg border p-4",
        highlight && value > 0
          ? "border-primary/20 bg-primary/5"
          : "border-border bg-card"
      )}
    >
      <span
        className={cn(
          "text-2xl font-bold tabular-nums",
          highlight && value > 0 ? "text-primary" : "text-foreground"
        )}
      >
        {value.toLocaleString()}
      </span>
      <span className="mt-1 text-xs text-muted-foreground">{label}</span>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface IngestSummaryProps {
  job: IngestJobStatus
  fileName: string
  onReset: () => void
}

export function IngestSummary({ job, fileName, onReset }: IngestSummaryProps) {
  const result = job.result!
  const { summary, rows } = result

  const duration =
    job.completed_at && job.started_at
      ? (
          (new Date(job.completed_at).getTime() -
            new Date(job.started_at).getTime()) /
          1000
        ).toFixed(1)
      : null

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="size-5 text-emerald-600" />
            <h3 className="text-base font-semibold">Ingest Complete</h3>
          </div>
          <p className="mt-0.5 text-sm text-muted-foreground">
            {fileName}
            {duration && (
              <span className="ml-2 text-muted-foreground/60">· {duration}s</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <a href={getDownloadUrl(job.job_id)} download>
            <Button variant="outline" size="sm" className="gap-1.5">
              <Download className="size-3.5" />
              Download Results
            </Button>
          </a>
          <Button variant="ghost" size="sm" className="gap-1.5" onClick={onReset}>
            <RefreshCw className="size-3.5" />
            Upload Another
          </Button>
        </div>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
        <StatCard label="Total" value={summary.total} />
        <StatCard label="Auto Match" value={summary.AUTO_MATCH} highlight />
        <StatCard label="Review Queue" value={summary.REVIEW_QUEUE} />
        <StatCard label="Create New" value={summary.CREATE_NEW} />
        <StatCard label="Unchanged" value={summary.UNCHANGED} />
        <StatCard label="Alerts" value={summary.alerts} />
      </div>

      <Separator />

      {/* Rows table */}
      <div>
        <h4 className="mb-3 text-sm font-medium">
          Row Details
          <span className="ml-2 text-xs font-normal text-muted-foreground">
            ({rows.length} row{rows.length !== 1 ? "s" : ""})
          </span>
        </h4>

        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full min-w-[700px] text-xs">
            <thead>
              <tr className="border-b border-border bg-muted/50">
                {[
                  "SKU ID",
                  "Brand",
                  "Package Type",
                  "Action",
                  "Matched SKU",
                  "Confidence",
                  "Signals",
                ].map((col) => (
                  <th
                    key={col}
                    className="px-3 py-2.5 text-left font-medium text-muted-foreground"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((row) => (
                <tr
                  key={row.tenant_sku_id}
                  className="bg-background transition-colors hover:bg-muted/30"
                >
                  <td className="px-3 py-2.5 font-mono text-[11px] text-muted-foreground">
                    {row.tenant_sku_id}
                  </td>
                  <td className="px-3 py-2.5 font-medium">{row.brand}</td>
                  <td className="max-w-[200px] px-3 py-2.5">
                    <p
                      className="truncate text-muted-foreground"
                      title={row.package_type ?? row.product_description}
                    >
                      {row.package_type ?? row.product_description}
                    </p>
                  </td>
                  <td className="px-3 py-2.5">
                    <ActionBadge action={row.action} />
                  </td>
                  <td className="px-3 py-2.5 font-mono text-[11px] text-muted-foreground">
                    {row.matched_global_sku ?? "—"}
                  </td>
                  <td className="px-3 py-2.5">
                    <ConfidencePill value={row.confidence} />
                  </td>
                  <td className="px-3 py-2.5">
                    <div className="flex flex-wrap gap-1">
                      {row.match_signals.map((sig) => (
                        <span
                          key={sig}
                          className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
                        >
                          {sig}
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Reasoning accordion for each row */}
        <details className="mt-3">
          <summary className="cursor-pointer select-none text-xs text-muted-foreground hover:text-foreground">
            Show match reasoning for all rows
          </summary>
          <div className="mt-2 flex flex-col gap-2">
            {rows.map((row) => (
              <div
                key={`reasoning-${row.tenant_sku_id}`}
                className="rounded-lg border border-border bg-muted/30 px-3 py-2"
              >
                <p className="mb-0.5 text-[11px] font-medium">
                  {row.tenant_sku_id} · {row.brand}
                </p>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {row.reasoning}
                </p>
              </div>
            ))}
          </div>
        </details>
      </div>
    </div>
  )
}
