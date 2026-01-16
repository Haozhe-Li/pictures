"use client"

import { motion } from "framer-motion"
import type { GalleryImage } from "@/types/gallery"
import { Loader2, Search } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

interface ImageGalleryProps {
  images: GalleryImage[]
  onImageClick: (image: GalleryImage) => void
  isSearching?: boolean
  staggerDelay?: number
  showMatch?: boolean
}

export function ImageGallery({ images, onImageClick, isSearching, staggerDelay = 0.05, showMatch = false }: ImageGalleryProps) {
  const [landscapeMap, setLandscapeMap] = useState<Record<string, boolean>>({})
  const [columnCount, setColumnCount] = useState(1)

  useEffect(() => {
    const getColumnCount = () => {
      if (typeof window === "undefined") {
        return 1
      }

      const width = window.innerWidth
      if (width >= 1280) return 4
      if (width >= 1024) return 3
      if (width >= 640) return 2
      return 2 // Use 2 columns on mobile for better viewing
    }

    const updateColumns = () => setColumnCount(getColumnCount())
    updateColumns()
    window.addEventListener("resize", updateColumns)
    return () => window.removeEventListener("resize", updateColumns)
  }, [])

  const orderedImages = useMemo(() => {
    if (!images || images.length === 0) {
      return images
    }

    const columns = Math.max(1, columnCount)
    const rows = Math.ceil(images.length / columns)
    const reordered: GalleryImage[] = []

    for (let col = 0; col < columns; col += 1) {
      for (let row = 0; row < rows; row += 1) {
        const index = row * columns + col
        if (index < images.length) {
          reordered.push(images[index])
        }
      }
    }

    return reordered
  }, [images, columnCount])


  if (isSearching) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <Loader2 className="w-10 h-10 animate-spin text-white/30" />
        <p className="text-white/40 text-sm">Searching...</p>
      </div>
    )
  }

  if (!images || !Array.isArray(images) || images.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-center justify-center h-96 gap-4"
      >
        <div className="w-20 h-20 rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
          <Search className="w-8 h-8 text-white/20" />
        </div>
        <div className="text-center">
          <p className="text-lg text-white/60 mb-1">No images found</p>
          <p className="text-sm text-white/30">Try a different search query</p>
        </div>
      </motion.div>
    )
  }

  return (
    <div className="columns-2 sm:columns-2 lg:columns-3 xl:columns-4 gap-3 sm:gap-6 space-y-3 sm:space-y-6">
      {orderedImages.map((image, index) => {
        const imageKey = image.preview_url || `image-${index}`
        const isLandscape = Boolean(landscapeMap[imageKey])
        const matchPercent = typeof image.score === "number" && !Number.isNaN(image.score)
          ? Math.round(image.score * 100)
          : null

        return (
          <motion.div
            key={`${image.preview_url}-${index}`}
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.6,
              delay: index * staggerDelay,
              ease: [0.25, 0.4, 0.25, 1],
            }}
            className="break-inside-avoid"
            layout
          >
            <div
              className="group relative overflow-hidden rounded-xl sm:rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 cursor-pointer 
                     hover:border-white/20 transition-transform duration-500"
              onClick={() => onImageClick(image)}
            >
              {/* Image */}
              <div className="relative overflow-hidden">
                <motion.img
                  src={image.preview_url || "/placeholder.svg"}
                  alt={image.metadata?.title || "Gallery image"}
                  className="w-full h-auto object-cover origin-center"
                  onLoad={(event) => {
                    const img = event.currentTarget
                    const landscape = img.naturalWidth > img.naturalHeight
                    setLandscapeMap((prev) => (prev[imageKey] === landscape ? prev : { ...prev, [imageKey]: landscape }))
                  }}
                  whileHover={{ scale: 1.02 }}
                  transition={{ duration: 0.6, ease: [0.25, 0.4, 0.25, 1] }}
                />

                {/* Gradient overlay */}
                <div
                  className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent 
                          opacity-0 group-hover:opacity-100 transition-all duration-500"
                />

                {/* Content on hover */}
                <div
                  className="absolute inset-0 flex flex-col justify-end p-3 sm:p-6 
                          opacity-0 group-hover:opacity-100 transition-all duration-500 translate-y-4 group-hover:translate-y-0"
                >
                  {image.metadata?.title && (
                    <h3 className="text-white font-semibold text-sm sm:text-xl mb-1 sm:mb-2 line-clamp-2">{image.metadata.title}</h3>
                  )}
                  {image.metadata?.description && (
                    <p className="text-white/80 text-xs sm:text-sm line-clamp-2 sm:line-clamp-3 leading-relaxed hidden sm:block">{image.metadata.description}</p>
                  )}
                </div>
              </div>

              {/* Score badge */}
              {showMatch && matchPercent !== null && (
                <motion.div
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.3 + index * 0.05 }}
                  className="absolute top-2 right-2 sm:top-4 sm:right-4 px-2 py-1 sm:px-3 sm:py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-white/10 text-[10px] sm:text-xs font-medium text-white/90"
                >
                  {matchPercent}%
                </motion.div>
              )}

              {/* Shimmer effect on hover */}
              <div
                className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 
                        bg-gradient-to-r from-transparent via-white/10 to-transparent pointer-events-none"
              />
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}
