import { type NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${BACKEND_URL}/generate-random-query`, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      console.error("Backend generate-random-query error:", response.status);
      return NextResponse.json({ query: "" });
    }

    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      console.error("Non-JSON response from backend");
      return NextResponse.json({ query: "" });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Generate random query API error:", error);
    return NextResponse.json({ query: "" });
  }
}
