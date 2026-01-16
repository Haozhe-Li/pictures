import { GalleryView } from "@/components/gallery-view"
import type { GalleryImage } from "@/types/gallery"

export const revalidate = 60

async function fetchGalleryData() {
    const baseUrl =
        process.env.NEXT_PUBLIC_SITE_URL ||
        (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "http://localhost:3000")

    try {
        const response = await fetch(`${baseUrl}/api/gallery?limit=20`, {
            next: { revalidate: 60 },
        })

        if (!response.ok) {
            return { items: [], next_cursor: null, ok: false }
        }

        const data = await response.json()
        return {
            items: Array.isArray(data.items) ? (data.items as GalleryImage[]) : [],
            next_cursor: data.next_cursor ?? null,
            ok: true,
        }
    } catch {
        return { items: [], next_cursor: null, ok: false }
    }
}

export default async function HomePage() {
    const data = await fetchGalleryData()
    const initialLoaded = data.ok

    return (
        <GalleryView
            initialImages={data.items}
            initialNextCursor={data.next_cursor}
            initialLoaded={initialLoaded}
        />
    )
}
