import type { QueryRequest, QueryResponse } from "@/types/chat"

export class ChatApiError extends Error {
  constructor(
    message: string,
    public readonly status: number
  ) {
    super(message)
    this.name = "ChatApiError"
  }
}

export async function askQuestion(req: QueryRequest): Promise<QueryResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  })

  const data = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))

  if (!res.ok) {
    throw new ChatApiError(
      data?.detail ?? `Request failed with status ${res.status}`,
      res.status
    )
  }

  return data as QueryResponse
}
