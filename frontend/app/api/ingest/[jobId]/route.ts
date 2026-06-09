import { NextResponse } from "next/server"

const API_BASE = process.env.API_BASE_URL ?? "http://localhost:8000"

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params

  try {
    const upstream = await fetch(`${API_BASE}/tenant/ingest/${jobId}`, {
      signal: AbortSignal.timeout(15_000),
    })

    const data = await upstream.json()
    return NextResponse.json(data, { status: upstream.status })
  } catch (err) {
    const isConnRefused =
      err instanceof Error &&
      (err.message.includes("ECONNREFUSED") || err.message.includes("fetch failed"))

    return NextResponse.json(
      {
        detail: isConnRefused
          ? "Could not reach the backend server."
          : "Failed to fetch job status.",
      },
      { status: 503 }
    )
  }
}
