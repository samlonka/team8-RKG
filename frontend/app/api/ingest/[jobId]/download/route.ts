import { NextResponse } from "next/server"

const API_BASE = process.env.API_BASE_URL ?? "http://localhost:8000"

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params

  try {
    const upstream = await fetch(
      `${API_BASE}/tenant/ingest/${jobId}/download`,
      { signal: AbortSignal.timeout(30_000) }
    )

    if (!upstream.ok) {
      const err = await upstream.json().catch(() => ({ detail: "Download failed" }))
      return NextResponse.json(err, { status: upstream.status })
    }

    // Stream the binary Excel file back to the browser
    const blob = await upstream.blob()
    const contentDisposition =
      upstream.headers.get("content-disposition") ??
      `attachment; filename="${jobId}_output.xlsx"`

    return new Response(blob, {
      status: 200,
      headers: {
        "Content-Type":
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "Content-Disposition": contentDisposition,
      },
    })
  } catch {
    return NextResponse.json({ detail: "Download failed." }, { status: 503 })
  }
}
