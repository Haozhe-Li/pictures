"use client"

import { useState, useCallback } from "react"
import { Search, X } from "lucide-react"
import useSWR from "swr"
import type { GalleryImage, GalleryResponse } from "@/lib/types"
import { SearchBar } from "./search-bar"
import { ImageGrid } from "./image-grid"
import { ImageDetailModal } from "./image-detail-modal"

const fetcher = (url: string) => fetch(url).then((res) => res.json())

export function Gallery() {
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<GalleryImage[] | null>(null)
  const [searchMeta, setSearchMeta] = useState<
    | { type: "text"; query: string }
    | { type: "similar"; title: string }
    | null
  >(null)
  const [isSearching, setIsSearching] = useState(false)
  const [selectedImage, setSelectedImage] = useState<GalleryImage | null>(null)
  const [isMobileSearchOpen, setIsMobileSearchOpen] = useState(false)

  const { data: galleryData, isLoading: isLoadingGallery } = useSWR<GalleryResponse>("/api/gallery", fetcher, {
    revalidateOnFocus: false,
  })

  const handleSearch = useCallback(async (query: string) => {
    window.scrollTo({ top: 0, behavior: "smooth" })

    setIsSearching(true)
    setSearchMeta({ type: "text", query })
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 50 }),
      })
      const data = await res.json()
      setSearchResults(data)
    } catch (error) {
      console.error("Search failed:", error)
    } finally {
      setIsSearching(false)
    }
  }, [])

  const handleClear = () => {
    setSearchQuery("")
    setSearchResults(null)
    setSearchMeta(null)
  }

  const handleSimilarSearch = useCallback(async (title: string, imageUrl: string) => {
    window.scrollTo({ top: 0, behavior: "smooth" })

    setIsSearching(true)
    setSearchQuery("")
    setSearchMeta({ type: "similar", title })
    try {
      const res = await fetch("/api/similar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_url: imageUrl, limit: 50 }),
      })
      const data = await res.json()
      setSearchResults(data)
    } catch (error) {
      console.error("Similar search failed:", error)
    } finally {
      setIsSearching(false)
    }
  }, [])

  const displayImages = searchResults ?? galleryData?.items ?? []
  const isLoading = isLoadingGallery || isSearching

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 bg-background">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="relative flex items-center justify-between h-16 sm:h-20 border-b border-border/40">
            {/* Logo / Site name (desktop + mobile collapsed) */}
            <a
              href="/"
              className={`font-serif text-xl sm:text-2xl tracking-tight transition-opacity duration-200 ${isMobileSearchOpen ? "opacity-0 pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 md:static md:translate-y-0 md:opacity-100 md:pointer-events-auto" : "opacity-100"}`}
            >
              Haozhe Li
            </a>

            {/* Mobile search toggle */}
            <div className="md:hidden flex items-center flex-1">
              <div
                className={`flex-1 overflow-hidden transition-opacity duration-200 ease-out ${isMobileSearchOpen
                  ? "opacity-100 max-w-full delay-200"
                  : "opacity-0 max-w-0 pointer-events-none"
                  }`}
              >
                <SearchBar
                  value={searchQuery}
                  onChange={setSearchQuery}
                  onSearch={handleSearch}
                  onClear={handleClear}
                  isLoading={isSearching}
                  className="w-full"
                />
              </div>

              <div className="ml-auto flex items-center justify-end pl-2">
                {isMobileSearchOpen ? (
                  <button
                    type="button"
                    onClick={() => setIsMobileSearchOpen(false)}
                    className="p-2 text-muted-foreground hover:text-foreground transition-colors"
                    aria-label="Close search"
                  >
                    <X className="h-5 w-5" />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => setIsMobileSearchOpen(true)}
                    className="p-2 text-muted-foreground hover:text-foreground transition-colors"
                    aria-label="Open search"
                  >
                    <Search className="h-5 w-5" />
                  </button>
                )}
              </div>
            </div>

            {/* Search - hidden on mobile, visible on larger screens */}
            <div className="hidden md:block flex-1 max-w-md mx-8">
              <SearchBar
                value={searchQuery}
                onChange={setSearchQuery}
                onSearch={handleSearch}
                onClear={handleClear}
                isLoading={isSearching}
              />
            </div>

            {/* Placeholder for future nav items */}
            <div className="hidden md:block w-20 sm:w-24" />
          </div>
        </div>
      </header>

      {/* Gallery */}
      <main className="px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {searchResults && searchMeta && (
          <div className="mb-4 text-sm sm:text-base text-muted-foreground leading-relaxed">
            {searchMeta.type === "text" ? (
              <span className="break-words">
                {searchResults.length} results for <span className="font-semibold text-foreground">{searchMeta.query}</span>
              </span>
            ) : (
              <span className="break-words">
                {searchResults.length} results, similar to <span className="font-semibold text-foreground">{searchMeta.title}</span> image
              </span>
            )}
          </div>
        )}
        <ImageGrid images={displayImages} onImageClick={setSelectedImage} isLoading={isLoading} />
      </main>

      <footer className="border-t border-border/40 py-6">
        <div className="px-4 sm:px-6 lg:px-8 text-sm text-muted-foreground text-center">
          © 2026{' '}
          <a
            href="https://haozhe.li"
            className="hover:text-foreground hover:underline underline-offset-4 transition-colors"
            target="_blank"
            rel="noreferrer"
          >
            Haozhe Li
          </a>
          . All rights reserved.
        </div>
      </footer>

      {/* Image Detail Modal */}
      <ImageDetailModal
        image={selectedImage}
        onClose={() => setSelectedImage(null)}
        onSimilarSearch={handleSimilarSearch}
      />
    </div>
  )
}
