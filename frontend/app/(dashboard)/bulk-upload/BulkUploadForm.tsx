"use client"

import { useState, useRef } from "react"
import { CloudUpload, X, FileText, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface UploadFile {
  id: string
  file: File
  progress: number
  status: "pending" | "uploading" | "done" | "error"
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function BulkUploadForm() {
  const [files, setFiles] = useState<UploadFile[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function addFiles(incoming: FileList | null) {
    if (!incoming) return
    const newEntries: UploadFile[] = Array.from(incoming).map((file) => ({
      id: `${file.name}-${Date.now()}-${Math.random()}`,
      file,
      progress: 0,
      status: "pending",
    }))
    setFiles((prev) => [...prev, ...newEntries])
  }

  function removeFile(id: string) {
    setFiles((prev) => prev.filter((f) => f.id !== id))
  }

  async function handleUpload() {
    const pending = files.filter((f) => f.status === "pending")
    if (!pending.length) return

    for (const entry of pending) {
      setFiles((prev) =>
        prev.map((f) =>
          f.id === entry.id ? { ...f, status: "uploading", progress: 0 } : f
        )
      )

      // Simulate upload progress
      for (let p = 10; p <= 100; p += 10) {
        await new Promise((r) => setTimeout(r, 120))
        setFiles((prev) =>
          prev.map((f) =>
            f.id === entry.id ? { ...f, progress: p } : f
          )
        )
      }

      setFiles((prev) =>
        prev.map((f) =>
          f.id === entry.id ? { ...f, status: "done", progress: 100 } : f
        )
      )
    }
  }

  const hasPending = files.some((f) => f.status === "pending")
  const isUploading = files.some((f) => f.status === "uploading")

  return (
    <div className="flex flex-col gap-5">
      {/* Drop zone */}
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          addFiles(e.dataTransfer.files)
        }}
        className={cn(
          "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-12 transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-border bg-muted/30 hover:border-primary/50 hover:bg-muted/50"
        )}
        aria-label="Upload files"
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
            Drop files here, or{" "}
            <span className="text-primary">browse to upload</span>
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            CSV, JSON, XLSX, TXT — up to 100 MB each
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".csv,.json,.xlsx,.xls,.txt"
          className="sr-only"
          onChange={(e) => addFiles(e.target.files)}
        />
      </button>

      {/* File list */}
      {files.length > 0 && (
        <ul className="flex flex-col gap-2">
          {files.map(({ id, file, progress, status }) => (
            <li
              key={id}
              className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-3"
            >
              <FileText className="size-4 shrink-0 text-muted-foreground" />
              <div className="flex-1 min-w-0">
                <p className="truncate text-sm font-medium">{file.name}</p>
                <div className="mt-1 flex items-center gap-2">
                  <p className="text-xs text-muted-foreground">
                    {formatBytes(file.size)}
                  </p>
                  {status === "uploading" && (
                    <>
                      <span className="text-xs text-muted-foreground">·</span>
                      <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-primary transition-all duration-150"
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {progress}%
                      </span>
                    </>
                  )}
                  {status === "done" && (
                    <span className="text-xs font-medium text-emerald-600">
                      · Uploaded
                    </span>
                  )}
                </div>
              </div>
              {status !== "uploading" && status !== "done" && (
                <button
                  type="button"
                  onClick={() => removeFile(id)}
                  className="shrink-0 text-muted-foreground transition-colors hover:text-destructive"
                  aria-label={`Remove ${file.name}`}
                >
                  <X className="size-4" />
                </button>
              )}
              {status === "uploading" && (
                <Loader2 className="size-4 shrink-0 animate-spin text-primary" />
              )}
            </li>
          ))}
        </ul>
      )}

      {/* Actions */}
      {files.length > 0 && (
        <div className="flex items-center gap-3">
          <Button
            onClick={handleUpload}
            disabled={!hasPending || isUploading}
            className="gap-2"
          >
            {isUploading ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                Uploading…
              </>
            ) : (
              <>
                <CloudUpload className="size-4" />
                Upload {hasPending ? `${files.filter((f) => f.status === "pending").length} file(s)` : ""}
              </>
            )}
          </Button>
          <Button
            variant="ghost"
            onClick={() => setFiles([])}
            disabled={isUploading}
          >
            Clear all
          </Button>
        </div>
      )}
    </div>
  )
}
