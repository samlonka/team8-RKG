import { NextResponse } from "next/server"
import type { QueryRequest } from "@/types/chat"

const API_BASE = process.env.API_BASE_URL ?? "http://localhost:8000"

export async function POST(request: Request) {
  let body: QueryRequest

  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 })
  }

  if (!body.question?.trim()) {
    return NextResponse.json({ detail: "question must not be empty" }, { status: 422 })
  }

  try {
    const upstream = await fetch(`${API_BASE}/query/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: body.question.trim(),
        anchor_sku: body.anchor_sku ?? null,
      }),
      // Generous timeout — LLM + graph traversal can take time
      signal: AbortSignal.timeout(120_000),
    })

    const data = await upstream.json()
    return NextResponse.json(data, { status: upstream.status })
  } catch (err) {
    const isTimeout = err instanceof Error && err.name === "TimeoutError"
    if (isTimeout) {
      return NextResponse.json(
        { detail: "The knowledge graph pipeline timed out. Try a simpler question." },
        { status: 504 }
      )
    }

    // Backend unreachable
    const isConnRefused =
      err instanceof Error &&
      (err.message.includes("ECONNREFUSED") || err.message.includes("fetch failed"))

    return NextResponse.json(
      {
        detail: isConnRefused
          ? "Could not connect to the backend server. Make sure it is running on port 8000."
          : "An unexpected error occurred while contacting the backend.",
      },
      { status: 503 }
    )
  }
}
