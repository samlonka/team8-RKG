"use client"

import { cn } from "@/lib/utils"
import type { PipelineEvent, QueryAskResponse } from "@/lib/rkg-api"
import { ChevronDown, Loader2 } from "lucide-react"
import { useState } from "react"

/** One-line status text from the latest pipeline event. */
export function getPipelineStatusLine(events: PipelineEvent[]): string {
  if (events.length === 0) return "Starting…"

  const latest = events[events.length - 1]
  const hits = latest.meta?.hits

  if (typeof hits === "number" && latest.status === "done") {
    return `${latest.title} · ${hits} hit${hits === 1 ? "" : "s"}`
  }

  return latest.title
}

interface PipelineStatusLineProps {
  events: PipelineEvent[]
  live?: boolean
  className?: string
}

/** Minimal single-line progress — like modern chat assistants. */
export function PipelineStatusLine({
  events,
  live = false,
  className,
}: PipelineStatusLineProps) {
  const text = getPipelineStatusLine(events)

  return (
    <div
      className={cn(
        "flex min-w-0 items-center gap-2 text-sm text-muted-foreground",
        className
      )}
    >
      {live ? (
        <Loader2 className="size-3.5 shrink-0 animate-spin text-primary" />
      ) : null}
      <span className="truncate">{text}</span>
    </div>
  )
}

const HIDDEN_STEP_TITLES = new Set([
  "Reading your question",
  "Designing the investigation plan",
  "Exploring the knowledge graph",
  "Quality-checking the evidence",
  "Re-checking the evidence",
  "Re-querying Neo4j",
  "Searching deeper",
])

function condensePipelineSteps(events: PipelineEvent[]): PipelineEvent[] {
  return events.filter(
    (ev) => ev.status === "done" && !HIDDEN_STEP_TITLES.has(ev.title)
  )
}

interface PipelineStepsCollapsibleProps {
  events: PipelineEvent[]
}

/** Optional expandable step log — hidden by default. */
export function PipelineStepsCollapsible({ events }: PipelineStepsCollapsibleProps) {
  const [open, setOpen] = useState(false)
  if (events.length === 0) return null

  const condensed = condensePipelineSteps(events)
  if (condensed.length === 0) return null

  return (
    <div className="mt-2 border-t border-border/60 pt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
      >
        <ChevronDown
          className={cn("size-3 transition-transform", open && "rotate-180")}
        />
        {open ? "Hide steps" : "Show steps"}
      </button>
      {open ? (
        <ul className="mt-2 space-y-1 text-[11px] text-muted-foreground">
          {condensed.map((ev) => (
            <li key={ev.id} className="truncate">
              <span className="text-foreground/70">{ev.title}</span>
              {ev.detail && ev.status === "done" && ev.detail.length < 80 ? (
                <span> — {ev.detail}</span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

interface ReasoningCollapsibleProps {
  reasoning: string
}

type DuplicateReport = NonNullable<QueryAskResponse["duplicate_report"]>

interface DuplicateReportCollapsibleProps {
  report: DuplicateReport
  defaultOpen?: boolean
}

/** Full duplicate group listing for catalog_duplicate answers. */
export function DuplicateReportCollapsible({
  report,
  defaultOpen = false,
}: DuplicateReportCollapsibleProps) {
  const [open, setOpen] = useState(defaultOpen)

  const upcGroups = report.upc_duplicate_groups ?? []
  const bpGroups = report.brand_package_duplicate_groups ?? []
  const totalListed = upcGroups.length + bpGroups.length
  const upcTotal = report.upc_groups_total ?? upcGroups.length
  const bpTotal = report.brand_package_groups_total ?? bpGroups.length
  const grandTotal = report.total_groups ?? upcTotal + bpTotal

  if (!report.has_duplicates || totalListed === 0) return null

  const listLabel =
    grandTotal > totalListed
      ? `Show ${totalListed} of ${grandTotal} duplicate group(s)`
      : `Show all ${totalListed} duplicate group(s)`

  return (
    <div className="mt-3 border-t border-border/60 pt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
      >
        <ChevronDown
          className={cn("size-3 transition-transform", open && "rotate-180")}
        />
        {open
          ? "Hide duplicate groups"
          : listLabel}
      </button>
      {open ? (
        <div className="mt-2 max-h-80 space-y-3 overflow-y-auto text-[11px]">
          {upcGroups.length > 0 ? (
            <section>
              <p className="mb-1 font-medium text-foreground/80">
                Duplicate UPC ({upcTotal > upcGroups.length ? `${upcGroups.length} of ${upcTotal}` : upcGroups.length})
              </p>
              <ul className="space-y-2 text-muted-foreground">
                {upcGroups.map((g) => (
                  <li key={g.upc ?? g.sku_ids.join("-")} className="rounded-md bg-background/60 px-2 py-1.5">
                    <span className="font-mono text-foreground/90">{g.upc}</span>
                    <span className="text-muted-foreground"> · {g.count} SKUs</span>
                    <p className="mt-0.5 break-all font-mono text-[10px]">
                      {g.sku_ids.join(", ")}
                    </p>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
          {bpGroups.length > 0 ? (
            <section>
              <p className="mb-1 font-medium text-foreground/80">
                Duplicate brand + package ({bpTotal > bpGroups.length ? `${bpGroups.length} of ${bpTotal}` : bpGroups.length})
              </p>
              <ul className="space-y-2 text-muted-foreground">
                {bpGroups.map((g) => (
                  <li
                    key={`${g.brand_name}-${g.package_type}`}
                    className="rounded-md bg-background/60 px-2 py-1.5"
                  >
                    <span className="text-foreground/90">
                      {g.brand_name} / {g.package_type}
                    </span>
                    <span className="text-muted-foreground"> · {g.count} SKUs</span>
                    <p className="mt-0.5 break-all font-mono text-[10px]">
                      {g.sku_ids.join(", ")}
                    </p>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

export function ReasoningCollapsible({ reasoning }: ReasoningCollapsibleProps) {
  const [open, setOpen] = useState(false)

  return (
    <div className="mt-3 border-t border-border/60 pt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
      >
        <ChevronDown
          className={cn("size-3 transition-transform", open && "rotate-180")}
        />
        {open ? "Hide reasoning" : "Show reasoning"}
      </button>
      {open ? (
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
          {reasoning}
        </p>
      ) : null}
    </div>
  )
}
