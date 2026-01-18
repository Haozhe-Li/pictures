"use client"

import { useEffect, useCallback } from "react"
import Image from "next/image"
import { X, Camera, Calendar } from "lucide-react"
import type { GalleryImage } from "@/lib/types"

interface ImageDetailModalProps {
  image: GalleryImage | null
  onClose: () => void
  onSimilarSearch: (imageTitle: string, imageUrl: string) => void
}

export function ImageDetailModal({ image, onClose, onSimilarSearch }: ImageDetailModalProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    },
    [onClose],
  )

  useEffect(() => {
    if (image) {
      document.body.style.overflow = "hidden"
      window.addEventListener("keydown", handleKeyDown)
    }
    return () => {
      document.body.style.overflow = ""
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [image, handleKeyDown])

  const handleViewFullRes = () => {
    if (!image) return
    window.open(image.original_url, "_blank")
  }

  const handleFindSimilar = () => {
    if (!image) return
    const title = image.metadata.title || "this"
    onSimilarSearch(title, image.original_url)
    onClose()
  }

  if (!image) return null

  const { metadata } = image

  return (
    <div className="fixed inset-0 z-50 bg-background overflow-y-auto">
      {/* Close button - fixed position */}
      <button
        onClick={onClose}
        className="fixed top-4 right-4 sm:top-6 sm:right-6 z-10 p-2 text-muted-foreground hover:text-foreground transition-colors"
        aria-label="Close"
      >
        <X className="h-5 w-5" />
      </button>

      {/* Main content - side by side on desktop, stacked on mobile */}
      <div className="min-h-screen flex flex-col lg:flex-row">
        {/* Image section */}
        <div className="flex-1 flex items-center justify-center p-4 pt-16 sm:p-8 sm:pt-20 lg:p-12">
          <Image
            src={image.original_url || "/placeholder.svg"}
            alt={metadata.title || "Gallery image"}
            width={1600}
            height={1200}
            className="max-w-full max-h-[50vh] lg:max-h-[85vh] w-auto h-auto object-contain"
            priority
          />
        </div>

        {/* Info section - compact and scrollable */}
        <div className="w-full lg:w-80 xl:w-96 p-6 sm:p-8 lg:py-20 lg:pr-12 lg:pl-0 flex flex-col justify-center">
          {/* Title */}
          <h1 className="font-serif text-2xl sm:text-3xl lg:text-2xl xl:text-3xl tracking-tight text-foreground">
            {metadata.title || "Untitled"}
          </h1>

          {/* Description */}
          {metadata.description && (
            <p className="mt-4 text-sm sm:text-base text-muted-foreground leading-relaxed">{metadata.description}</p>
          )}

          {/* Metadata */}
          <div className="mt-6 space-y-2 text-sm text-muted-foreground">
            {metadata.camera && (
              <div className="flex items-center gap-2">
                <Camera className="h-4 w-4 shrink-0" />
                <span>{metadata.camera}</span>
              </div>
            )}
            {metadata.taken_time && (
              <div className="flex items-center gap-2">
                <Calendar className="h-4 w-4 shrink-0" />
                <span>{metadata.taken_time} UTC</span>
              </div>
            )}
          </div>

          {/* Action buttons - small text links */}
          <div className="mt-8 flex items-center gap-6 text-sm">
            <button
              onClick={handleViewFullRes}
              className="text-muted-foreground hover:text-foreground underline underline-offset-4 transition-colors"
            >
              View Full-Res
            </button>
            <button
              onClick={handleFindSimilar}
              className="text-muted-foreground hover:text-foreground underline underline-offset-4 transition-colors"
            >
              Find Similar
            </button>
          </div>

          <div className="mt-8 text-xs text-muted-foreground">
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
        </div>
      </div>
    </div>
  )
}
