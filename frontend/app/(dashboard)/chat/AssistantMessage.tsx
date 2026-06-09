"use client"

import { useState } from "react"
import {
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Activity,
  Network,
  Zap,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { QueryResponse, ValidatedChain, EntityNode } from "@/types/chat"

// ── Classification badge ──────────────────────────────────────────────────────

const CLASSIFICATION_STYLES: Record<string, { className: string; icon: React.ReactNode }> = {
  "Confirmed Anomaly": {
    className: "bg-destructive/10 text-destructive border border-destructive/20",
    icon: <AlertTriangle className="size-3" />,
  },
  "Needs Review": {
    className: "bg-amber-500/10 text-amber-600 border border-amber-500/20",
    icon: <Clock className="size-3" />,
  },
  Healthy: {
    className: "bg-emerald-500/10 text-emerald-600 border border-emerald-500/20",
    icon: <CheckCircle2 className="size-3" />,
  },
}

function ClassificationBadge({ label }: { label: string }) {
  const style = CLASSIFICATION_STYLES[label] ?? CLASSIFICATION_STYLES["Needs Review"]
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium",
        style.className
      )}
    >
      {style.icon}
      {label}
    </span>
  )
}

// ── Confidence bar ────────────────────────────────────────────────────────────

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 80 ? "bg-emerald-500" : pct >= 60 ? "bg-amber-500" : "bg-destructive"
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full transition-all", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] font-medium tabular-nums">{pct}%</span>
    </div>
  )
}

// ── Entity path chip ──────────────────────────────────────────────────────────

function EntityChip({ node }: { node: EntityNode }) {
  const isHighAnomaly = (node.anomaly_score ?? 0) >= 0.7
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[10px] font-medium",
        isHighAnomaly
          ? "border-destructive/30 bg-destructive/5 text-destructive"
          : "border-border bg-muted/50 text-muted-foreground"
      )}
      title={node.entity_id}
    >
      <span className="opacity-60">{node.label}:</span>
      {node.display_name}
      {node.anomaly_score !== null && (
        <span
          className={cn(
            "ml-0.5 rounded px-0.5 text-[9px] tabular-nums",
            isHighAnomaly ? "text-destructive" : "text-muted-foreground/70"
          )}
        >
          {node.anomaly_score.toFixed(2)}
        </span>
      )}
    </span>
  )
}

// ── Single validated chain row ────────────────────────────────────────────────

function ChainRow({ chain, index }: { chain: ValidatedChain; index: number }) {
  const [open, setOpen] = useState(false)
  const VISIBLE = 5
  const visiblePath = chain.path.slice(0, VISIBLE)
  const hiddenCount = chain.path.length - VISIBLE

  const sourceLabel =
    chain.source === "union"
      ? "Cross-source"
      : chain.source === "ann_reflect"
        ? "ANN Reflect"
        : chain.source === "ann_self"
          ? "ANN Self"
          : "Cypher"

  return (
    <div className="rounded-lg border border-border bg-background/60 text-xs">
      {/* Header row */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 px-3 py-2 text-left transition-colors hover:bg-muted/40"
      >
        <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
          #{index + 1}
        </span>
        <ConfidenceBar value={chain.confidence} />
        <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
          {sourceLabel}
        </span>
        <span className="ml-auto text-[10px] text-muted-foreground">
          {chain.path.length} entities
        </span>
        {open ? (
          <ChevronDown className="size-3 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-3 shrink-0 text-muted-foreground" />
        )}
      </button>

      {/* Expanded detail */}
      {open && (
        <div className="border-t border-border px-3 py-3 space-y-3">
          {/* Entity path */}
          <div>
            <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              Entity Path
            </p>
            <div className="flex flex-wrap items-center gap-1.5">
              {visiblePath.map((node, i) => (
                <span key={node.entity_id} className="flex items-center gap-1.5">
                  <EntityChip node={node} />
                  {i < visiblePath.length - 1 && (
                    <span className="text-muted-foreground/50">→</span>
                  )}
                </span>
              ))}
              {hiddenCount > 0 && (
                <span className="text-[10px] text-muted-foreground">
                  +{hiddenCount} more
                </span>
              )}
            </div>
          </div>

          {/* Scores row */}
          <div className="flex flex-wrap gap-4">
            {[
              { label: "Temporal", value: chain.temporal_validity },
              { label: "Density", value: chain.evidence_density },
              { label: "Anomaly", value: chain.avg_anomaly_score },
            ].map(({ label, value }) => (
              <div key={label}>
                <p className="text-[10px] text-muted-foreground">{label}</p>
                <ConfidenceBar value={value} />
              </div>
            ))}
          </div>

          {/* Reasoning */}
          {chain.reasoning && (
            <div>
              <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                Reasoning
              </p>
              <p className="text-[11px] leading-relaxed text-foreground/80">{chain.reasoning}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Collapsible section ───────────────────────────────────────────────────────

function Section({
  title,
  icon,
  defaultOpen = false,
  children,
}: {
  title: string
  icon: React.ReactNode
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 bg-muted/30 px-3 py-2 text-left text-xs font-medium transition-colors hover:bg-muted/50"
      >
        {icon}
        <span>{title}</span>
        {open ? (
          <ChevronDown className="ml-auto size-3 text-muted-foreground" />
        ) : (
          <ChevronRight className="ml-auto size-3 text-muted-foreground" />
        )}
      </button>
      {open && <div className="p-3">{children}</div>}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface AssistantMessageProps {
  response: QueryResponse
  timestamp: Date
}

export function AssistantMessage({ response, timestamp }: AssistantMessageProps) {
  const {
    summary,
    best_confidence,
    best_classification,
    best_reasoning,
    planner_rationale,
    validated_chains,
    latency_seconds,
    tasks,
  } = response

  const hasChains = validated_chains.length > 0
  const hasReasoning = !!best_reasoning
  const hasPlanner = !!planner_rationale

  return (
    <div className="flex flex-col gap-2 max-w-[85%]">
      {/* Primary bubble */}
      <div className="rounded-xl bg-muted px-4 py-3 text-sm">
        {/* Meta row */}
        <div className="mb-2.5 flex flex-wrap items-center gap-2">
          {best_classification && (
            <ClassificationBadge label={best_classification} />
          )}
          {best_confidence !== null && best_confidence !== undefined && (
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-muted-foreground">Confidence</span>
              <ConfidenceBar value={best_confidence} />
            </div>
          )}
          {hasChains && (
            <span className="inline-flex items-center gap-1 text-[10px] text-muted-foreground">
              <Network className="size-3" />
              {validated_chains.length} chain{validated_chains.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>

        {/* Summary text — preserve line breaks from backend */}
        <p className="whitespace-pre-wrap leading-relaxed text-foreground">
          {summary}
        </p>

        {/* Timestamp + latency */}
        <div className="mt-2 flex items-center justify-between gap-2">
          <span className="text-[10px] text-muted-foreground">
            {timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
          {latency_seconds > 0 && (
            <span className="inline-flex items-center gap-1 text-[10px] text-muted-foreground">
              <Zap className="size-3" />
              {latency_seconds.toFixed(2)}s
            </span>
          )}
        </div>
      </div>

      {/* Expandable sections */}
      {(hasReasoning || hasChains || hasPlanner) && (
        <div className="flex flex-col gap-1.5 pl-1">
          {/* Best chain reasoning */}
          {hasReasoning && (
            <Section
              title="Chain Reasoning"
              icon={<Activity className="size-3 text-muted-foreground" />}
            >
              <p className="text-xs leading-relaxed text-foreground/80">{best_reasoning}</p>
            </Section>
          )}

          {/* Validated chains */}
          {hasChains && (
            <Section
              title={`Evidence Chains (${validated_chains.length})`}
              icon={<Network className="size-3 text-muted-foreground" />}
              defaultOpen={validated_chains.length === 1}
            >
              <div className="flex flex-col gap-2">
                {validated_chains.map((chain, i) => (
                  <ChainRow key={chain.chain_id} chain={chain} index={i} />
                ))}
              </div>
            </Section>
          )}

          {/* Planner rationale */}
          {hasPlanner && (
            <Section
              title={`Planner Rationale${tasks.length ? ` · ${tasks.length} tasks` : ""}`}
              icon={<Zap className="size-3 text-muted-foreground" />}
            >
              <p className="text-xs leading-relaxed text-foreground/80">{planner_rationale}</p>
              {tasks.length > 0 && (
                <ol className="mt-2 flex flex-col gap-1">
                  {tasks.map((t) => (
                    <li
                      key={t.step}
                      className="flex items-start gap-2 text-[11px] text-muted-foreground"
                    >
                      <span className="shrink-0 font-mono text-[10px]">{t.step}.</span>
                      <span>
                        <span className="rounded bg-muted px-1 py-0.5 font-medium text-foreground/70">
                          {t.task_type}
                        </span>{" "}
                        {t.description || t.label}
                      </span>
                    </li>
                  ))}
                </ol>
              )}
            </Section>
          )}
        </div>
      )}
    </div>
  )
}
