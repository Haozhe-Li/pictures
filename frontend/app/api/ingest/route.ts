import { type NextRequest, NextResponse } from "next/server"

export const runtime = "nodejs"

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get("authorization")

    if (!authHeader) {
      return NextResponse.json({ error: "Authentication required" }, { status: 401 })
    }

    // Get the form data from the request
    const formData = await request.formData()

    // Forward the request to the backend
    const backendUrl = "http://127.0.0.1:8000/ingest"

    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        Authorization: authHeader,
      },
      body: formData,
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json({ error: data.detail || "Upload failed" }, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Upload error:", error)
    return NextResponse.json({ error: "Failed to upload image" }, { status: 500 })
  }
}
