import { NextResponse } from "next/server"
import { getRandomQuery } from "@/lib/api"

export async function GET() {
  try {
    const query = await getRandomQuery()
    return NextResponse.json({ query })
  } catch (error) {
    console.error("Random query error:", error)
    return NextResponse.json({ error: "Failed to generate random query" }, { status: 500 })
  }
}
