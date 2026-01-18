import { NextResponse } from "next/server"
import { getSimilarImages } from "@/lib/api"

export async function POST(request: Request) {
  try {
    const { image_url, limit = 8 } = await request.json()

    if (!image_url) {
      return NextResponse.json({ error: "Image URL is required" }, { status: 400 })
    }

    const data = await getSimilarImages(image_url, limit)
    return NextResponse.json(data)
  } catch (error) {
    console.error("Similar images error:", error)
    return NextResponse.json({ error: "Failed to get similar images" }, { status: 500 })
  }
}
