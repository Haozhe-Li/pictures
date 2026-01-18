import { NextResponse } from "next/server"
import { getGallery } from "@/lib/api"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = Number.parseInt(searchParams.get("limit") || "50")
    const cursor = searchParams.get("cursor") || undefined

    const data = await getGallery(limit, cursor)
    return NextResponse.json(data)
  } catch (error) {
    console.error("Gallery fetch error:", error)
    return NextResponse.json({ error: "Failed to fetch gallery" }, { status: 500 })
  }
}
