import type { IngestJobCreated, IngestJobStatus } from "@/types/ingest"

export class IngestApiError extends Error {
  constructor(
    message: string,
    public readonly status: number
  ) {
    super(message)
    this.name = "IngestApiError"
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  const data = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
  if (!res.ok) {
    throw new IngestApiError(data?.detail ?? `Request failed (${res.status})`, res.status)
  }
  return data as T
}

/** Upload an Excel file to start an ingest job. */
export async function uploadFile(file: File): Promise<IngestJobCreated> {
  const body = new FormData()
  body.append("file", file)

  const res = await fetch("/api/ingest", { method: "POST", body })
  return handleResponse<IngestJobCreated>(res)
}

/** Poll the status of an ingest job. */
export async function getJobStatus(jobId: string): Promise<IngestJobStatus> {
  const res = await fetch(`/api/ingest/${jobId}`)
  return handleResponse<IngestJobStatus>(res)
}

/** Returns the URL to download the annotated output Excel. */
export function getDownloadUrl(jobId: string): string {
  return `/api/ingest/${jobId}/download`
}
