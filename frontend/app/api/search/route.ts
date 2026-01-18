import { NextResponse } from "next/server"
import { searchImages } from "@/lib/api"

export async function POST(request: Request) {
  try {
    const { query, limit = 20 } = await request.json()

    if (!query) {
      return NextResponse.json({ error: "Query is required" }, { status: 400 })
    }

    const data = await searchImages(query, limit)
    return NextResponse.json(data)
  } catch (error) {
    console.error("Search error:", error)
    return NextResponse.json({ error: "Failed to search images" }, { status: 500 })
  }
}
