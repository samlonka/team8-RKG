"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import {
  CloudUpload,
  X,
  FileSpreadsheet,
  Loader2,
  AlertCircle,
  RefreshCw,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { uploadFile, getJobStatus, IngestApiError } from "@/services/ingestApi"
import { IngestSummary } from "./IngestSummary"
import type { IngestJobStatus } from "@/types/ingest"

// ── Upload phase state machine ────────────────────────────────────────────────

type Phase =
  | { kind: "idle" }
  | { kind: "selected"; file: File }
  | { kind: "uploading"; file: File }
  | { kind: "processing"; file: File; jobId: string; pollCount: number }
  | { kind: "completed"; file: File; job: IngestJobStatus }
  | { kind: "error"; file: File | null; message: string }

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const POLL_INTERVAL_MS = 5_000

// ── Component ─────────────────────────────────────────────────────────────────

export function BulkUploadForm() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" })
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Polling ──────────────────────────────────────────────────────────────

  const stopPolling = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [])

  useEffect(() => {
    if (phase.kind !== "processing") {
      stopPolling()
      return
    }

    const { jobId, file } = phase

    async function poll() {
      try {
        const status = await getJobStatus(jobId)

        if (status.status === "completed") {
          stopPolling()
          setPhase({ kind: "completed", file, job: status })
          return
        }

        if (status.status === "failed") {
          stopPolling()
          setPhase({
            kind: "error",
            file,
            message: status.error ?? "Ingest job failed on the server.",
          })
          return
        }

        // Still pending/processing — increment counter for UI feedback
        setPhase((prev) =>
          prev.kind === "processing"
            ? { ...prev, pollCount: prev.pollCount + 1 }
            : prev
        )
      } catch {
        // Network blip — keep polling; don't surface transient errors
      }
    }

    // Immediate first check, then every 5 s
    poll()
    intervalRef.current = setInterval(poll, POLL_INTERVAL_MS)

    return stopPolling
  }, [phase.kind === "processing" ? phase.jobId : null, stopPolling]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── File selection ────────────────────────────────────────────────────────

  function selectFile(file: File | undefined) {
    if (!file) return
    if (!file.name.match(/\.(xlsx|xls)$/i)) {
      setPhase({ kind: "error", file: null, message: "Only Excel files (.xlsx, .xls) are accepted." })
      return
    }
    setPhase({ kind: "selected", file })
  }

  function handleDrop(e: React.DragEvent<HTMLButtonElement>) {
    e.preventDefault()
    setIsDragging(false)
    selectFile(e.dataTransfer.files[0])
  }

  // ── Upload ────────────────────────────────────────────────────────────────

  async function handleUpload() {
    if (phase.kind !== "selected") return
    const { file } = phase

    setPhase({ kind: "uploading", file })

    try {
      const created = await uploadFile(file)
      setPhase({ kind: "processing", file, jobId: created.job_id, pollCount: 0 })
    } catch (err) {
      const message =
        err instanceof IngestApiError ? err.message : "Upload failed. Please try again."
      setPhase({ kind: "error", file, message })
    }
  }

  // ── Reset ─────────────────────────────────────────────────────────────────

  function reset() {
    stopPolling()
    setPhase({ kind: "idle" })
    if (inputRef.current) inputRef.current.value = ""
  }

  // ── Render ────────────────────────────────────────────────────────────────

  // Completed — show full summary
  if (phase.kind === "completed") {
    return (
      <IngestSummary
        job={phase.job}
        fileName={phase.file.name}
        onReset={reset}
      />
    )
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Drop zone — hidden while processing or uploading */}
      {(phase.kind === "idle" || phase.kind === "selected") && (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={cn(
            "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-12 transition-colors",
            isDragging
              ? "border-primary bg-primary/5"
              : "border-border bg-muted/30 hover:border-primary/50 hover:bg-muted/50"
          )}
          aria-label="Upload Excel file"
        >
          <div
            className={cn(
              "flex size-12 items-center justify-center rounded-full transition-colors",
              isDragging ? "bg-primary/15" : "bg-muted"
            )}
          >
            <CloudUpload
              className={cn(
                "size-6 transition-colors",
                isDragging ? "text-primary" : "text-muted-foreground"
              )}
            />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium">
              Drop your Excel file here, or{" "}
              <span className="text-primary">browse to upload</span>
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Accepts .xlsx and .xls files
            </p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".xlsx,.xls"
            className="sr-only"
            onChange={(e) => selectFile(e.target.files?.[0])}
          />
        </button>
      )}

      {/* Selected file */}
      {phase.kind === "selected" && (
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-3">
          <FileSpreadsheet className="size-5 shrink-0 text-emerald-600" />
          <div className="flex-1 min-w-0">
            <p className="truncate text-sm font-medium">{phase.file.name}</p>
            <p className="text-xs text-muted-foreground">{formatBytes(phase.file.size)}</p>
          </div>
          <button
            type="button"
            onClick={reset}
            className="shrink-0 text-muted-foreground transition-colors hover:text-destructive"
            aria-label="Remove file"
          >
            <X className="size-4" />
          </button>
        </div>
      )}

      {/* Uploading state */}
      {phase.kind === "uploading" && (
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-4">
          <FileSpreadsheet className="size-5 shrink-0 text-primary" />
          <div className="flex-1 min-w-0">
            <p className="truncate text-sm font-medium">{phase.file.name}</p>
            <div className="mt-1.5 flex items-center gap-2">
              <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
                <div className="h-full w-1/2 animate-pulse rounded-full bg-primary" />
              </div>
              <span className="text-xs text-muted-foreground">Uploading…</span>
            </div>
          </div>
          <Loader2 className="size-4 shrink-0 animate-spin text-primary" />
        </div>
      )}

      {/* Processing / polling state */}
      {phase.kind === "processing" && (
        <div className="flex flex-col items-center gap-4 rounded-xl border border-border bg-muted/30 px-6 py-10">
          <div className="relative flex size-14 items-center justify-center">
            <div className="absolute inset-0 animate-ping rounded-full bg-primary/20" />
            <div className="relative flex size-14 items-center justify-center rounded-full bg-primary/10">
              <Loader2 className="size-6 animate-spin text-primary" />
            </div>
          </div>
          <div className="text-center">
            <p className="text-sm font-semibold">Processing your data…</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Matching SKUs against the Global Knowledge Graph
            </p>
            <p className="mt-2 font-mono text-[10px] text-muted-foreground/60">
              Job {phase.jobId.slice(0, 8)}… · Checking every 5s
              {phase.pollCount > 0 && ` · ${phase.pollCount} poll${phase.pollCount !== 1 ? "s" : ""}`}
            </p>
          </div>
        </div>
      )}

      {/* Error state */}
      {phase.kind === "error" && (
        <div className="flex items-start gap-3 rounded-lg border border-destructive/20 bg-destructive/5 px-4 py-3">
          <AlertCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
          <div className="flex-1">
            <p className="text-sm font-medium text-destructive">Upload failed</p>
            <p className="mt-0.5 text-xs text-destructive/80">{phase.message}</p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={reset}
            className="shrink-0 gap-1.5 text-destructive hover:text-destructive"
          >
            <RefreshCw className="size-3.5" />
            Retry
          </Button>
        </div>
      )}

      {/* Upload action */}
      {phase.kind === "selected" && (
        <div className="flex items-center gap-3">
          <Button onClick={handleUpload} className="gap-2">
            <CloudUpload className="size-4" />
            Upload &amp; Process
          </Button>
          <Button variant="ghost" onClick={reset}>
            Cancel
          </Button>
        </div>
      )}
    </div>
  )
}
