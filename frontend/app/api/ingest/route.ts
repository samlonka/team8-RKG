import { NextResponse } from "next/server"

const API_BASE = process.env.API_BASE_URL ?? "http://localhost:8000"

export async function POST(request: Request) {
  let formData: FormData

  try {
    formData = await request.formData()
  } catch {
    return NextResponse.json({ detail: "Invalid form data" }, { status: 400 })
  }

  const file = formData.get("file")
  if (!file || !(file instanceof File)) {
    return NextResponse.json({ detail: "No file provided" }, { status: 422 })
  }

  try {
    // Forward the multipart form exactly as-is — fetch sets the boundary automatically
    const upstream = await fetch(
      `${API_BASE}/tenant/ingest?skip_validation=false`,
      {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(30_000),
      }
    )

    const data = await upstream.json()
    return NextResponse.json(data, { status: upstream.status })
  } catch (err) {
    const isConnRefused =
      err instanceof Error &&
      (err.message.includes("ECONNREFUSED") || err.message.includes("fetch failed"))

    return NextResponse.json(
      {
        detail: isConnRefused
          ? "Could not connect to the backend. Make sure it is running on port 8000."
          : "Upload failed due to an unexpected server error.",
      },
      { status: 503 }
    )
  }
}
