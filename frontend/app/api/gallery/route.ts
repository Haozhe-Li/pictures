import { type NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = searchParams.get("limit") || "20";
    const cursor = searchParams.get("cursor");

    const url = new URL(`${BACKEND_URL}/gallery`);
    url.searchParams.append("limit", limit);
    if (cursor) {
      url.searchParams.append("cursor", cursor);
    }

    const response = await fetch(url.toString(), {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      next: {
        revalidate: 60, // Cache for 60 seconds to handle backend cold starts
      },
    });

    if (!response.ok) {
      console.error(`Backend error: ${response.status} ${response.statusText}`);
      return NextResponse.json(
        { items: [], next_cursor: null },
        { status: 200 }
      );
    }

    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      const text = await response.text();
      console.error("Non-JSON response from backend:", text.substring(0, 200));
      return NextResponse.json(
        { items: [], next_cursor: null },
        { status: 200 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "public, s-maxage=60, stale-while-revalidate=300",
      },
    });
  } catch (error) {
    console.error(
      "Gallery API error:",
      error instanceof Error ? error.message : error
    );
    return NextResponse.json({ items: [], next_cursor: null }, { status: 200 });
  }
}
