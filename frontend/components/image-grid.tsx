"use client"

import { useState, useEffect } from "react"
import Image from "next/image"
import type { GalleryImage } from "@/lib/types"
import { cn } from "@/lib/utils"

interface ImageGridProps {
  images: GalleryImage[]
  onImageClick: (image: GalleryImage) => void
  isLoading?: boolean
}

function ImageSkeleton({ index }: { index: number }) {
  // Vary heights for visual interest
  const heights = [280, 320, 240, 360, 300, 260, 340, 280, 320, 240, 300, 260]
  const height = heights[index % heights.length]

  return (
    <div className="break-inside-avoid rounded-lg overflow-hidden" style={{ height: `${height}px` }}>
      <div className="w-full h-full bg-secondary animate-pulse" />
    </div>
  )
}

export function ImageGrid({ images, onImageClick, isLoading }: ImageGridProps) {
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set())
  const [visibleImages, setVisibleImages] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (!isLoading && images.length > 0) {
      // Reset visible images when new results come in
      setVisibleImages(new Set())

      // Stagger the appearance of each image
      images.forEach((image, index) => {
        setTimeout(() => {
          setVisibleImages((prev) => new Set(prev).add(image.original_url))
        }, index * 50) // 50ms stagger between each image
      })
    }
  }, [images, isLoading])

  if (isLoading) {
    return (
      <div className="columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-4 space-y-4">
        {Array.from({ length: 12 }).map((_, i) => (
          <ImageSkeleton key={i} index={i} />
        ))}
      </div>
    )
  }

  if (images.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <p className="text-muted-foreground text-lg">No images found</p>
        <p className="text-muted-foreground/60 text-sm mt-1">Try a different search term</p>
      </div>
    )
  }

  return (
    <div className="columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-4 space-y-4">
      {images.map((image, index) => (
        <button
          key={image.original_url}
          onClick={() => onImageClick(image)}
          className={cn(
            "break-inside-avoid block w-full group relative overflow-hidden rounded-lg bg-secondary",
            "transition-all duration-500 ease-out",
            "hover:shadow-xl hover:shadow-foreground/5",
            visibleImages.has(image.original_url) ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4",
          )}
        >
          <div
            className={cn(
              "relative w-full transition-opacity duration-500",
              loadedImages.has(image.preview_url) ? "opacity-100" : "opacity-0",
            )}
          >
            <Image
              src={image.preview_url || "/placeholder.svg"}
              alt={image.metadata.title || "Gallery image"}
              width={600}
              height={800}
              className="w-full h-auto object-cover"
              onLoad={() => setLoadedImages((prev) => new Set(prev).add(image.preview_url))}
              priority={index < 8}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-foreground/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <div className="absolute bottom-0 left-0 right-0 p-4 translate-y-4 opacity-0 group-hover:translate-y-0 group-hover:opacity-100 transition-all duration-300">
              <h3 className="text-background font-medium text-sm truncate">{image.metadata.title || "Untitled"}</h3>
              {image.metadata.camera && <p className="text-background/70 text-xs mt-0.5">{image.metadata.camera}</p>}
            </div>
          </div>
          {!loadedImages.has(image.preview_url) && <div className="absolute inset-0 animate-pulse bg-secondary" />}
        </button>
      ))}
    </div>
  )
}
