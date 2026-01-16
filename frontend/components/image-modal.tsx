"use client"

import { motion, AnimatePresence } from "framer-motion"
import { X, Calendar, Camera, ArrowUpRight, Sparkles, Loader2, Fullscreen } from "lucide-react"
import type { GalleryImage } from "@/types/gallery"
import { useEffect, useState } from "react"

interface ImageModalProps {
  image: GalleryImage | null
  onClose: () => void
  onSimilarTo?: (image: GalleryImage) => void
  isSimilarLoading?: boolean
  showMatch?: boolean
}

export function ImageModal({ image, onClose, onSimilarTo, isSimilarLoading = false, showMatch = false }: ImageModalProps) {
  const [isImageLoading, setIsImageLoading] = useState(false)

  const matchPercent = typeof image?.score === "number" && !Number.isNaN(image.score)
    ? Math.round(image.score * 100)
    : null


  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose()
      }
    }

    if (image) {
      document.addEventListener("keydown", handleEscape)
      document.body.style.overflow = "hidden"
    }

    return () => {
      document.removeEventListener("keydown", handleEscape)
      document.body.style.overflow = "unset"
    }
  }, [image, onClose])

  useEffect(() => {
    if (image) {
      setIsImageLoading(true)
    }
  }, [image])

  return (
    <AnimatePresence>
      {image && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="fixed inset-0 z-50 flex items-start lg:items-center justify-center p-3 pt-14 lg:p-8 bg-black/98 backdrop-blur-2xl overflow-y-auto"
          onClick={onClose}
        >
          {/* Background gradient effects */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
            <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />
          </div>

          {/* Close button */}
          <motion.button
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, type: "spring" }}
            onClick={onClose}
            className="absolute top-4 right-4 lg:top-6 lg:right-6 z-10 p-2 lg:p-3 rounded-full bg-white/10 hover:bg-white/20 border border-white/10 
                     hover:border-white/20 transition-all duration-300 group"
          >
            <X className="w-5 h-5 lg:w-6 lg:h-6 text-white/80 group-hover:text-white transition-colors" />
          </motion.button>

          {/* Content */}
          <motion.div
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            transition={{ duration: 0.4, type: "spring", damping: 25 }}
            className="relative max-w-7xl w-full max-h-[90vh] overflow-y-auto lg:overflow-visible"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Mobile Layout */}
            <div className="lg:hidden flex flex-col gap-4">
              {/* Image Container - Mobile */}
              <div className="relative flex items-center justify-center rounded-2xl bg-white/5 border border-white/10 backdrop-blur-sm overflow-hidden">
                {isImageLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                    <div className="flex items-center gap-3 text-white/60">
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span className="text-sm">Loading...</span>
                    </div>
                  </div>
                )}
                <motion.img
                  initial={{ scale: 1.1, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.6, ease: [0.25, 0.4, 0.25, 1] }}
                  src={image.original_url || "/placeholder.svg"}
                  alt={image.metadata?.title || "Gallery image"}
                  className="max-w-full max-h-[50vh] object-contain"
                  onLoad={() => setIsImageLoading(false)}
                  onError={() => setIsImageLoading(false)}
                />
              </div>

              {/* Action Buttons - Mobile (Top Priority) */}
              <div className="flex gap-2">
                <button
                  onClick={() => window.open(image.original_url, "_blank")}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 
                           border border-white/10 text-white text-sm font-medium transition-all"
                >
                  <Fullscreen className="w-4 h-4" />
                  <span>Full-Res</span>
                </button>
                <button
                  onClick={() => image && onSimilarTo?.(image)}
                  disabled={!onSimilarTo || isSimilarLoading}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl bg-purple-500/20 hover:bg-purple-500/30 
                           border border-purple-500/30 text-white text-sm font-medium transition-all
                           disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {isSimilarLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                  <span>Similar</span>
                </button>
              </div>

              {/* Details Panel - Mobile */}
              <div className="bg-white/5 backdrop-blur-2xl rounded-2xl p-4 border border-white/10">
                {/* Title & Match Badge */}
                <div className="flex items-start justify-between gap-3 mb-3">
                  {image.metadata?.title && (
                    <h2 className="text-xl font-bold text-white leading-tight flex-1">{image.metadata.title}</h2>
                  )}
                  {showMatch && matchPercent !== null && (
                    <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-white/10 border border-white/20 shrink-0">
                      <div className="w-1.5 h-1.5 rounded-full bg-white/60 animate-pulse" />
                      <span className="text-xs font-medium text-white/80">{matchPercent}%</span>
                    </div>
                  )}
                </div>

                {/* Description - Collapsible on mobile */}
                {image.metadata?.description && (
                  <p className="text-white/70 text-sm leading-relaxed mb-4 line-clamp-3">{image.metadata.description}</p>
                )}

                {/* Metadata - Compact Grid for Mobile */}
                {(image.metadata?.taken_time || image.metadata?.camera) && (
                  <div className="flex flex-wrap gap-2 pt-3 border-t border-white/10">
                    {image.metadata?.taken_time && (
                      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10">
                        <Calendar className="w-3.5 h-3.5 text-white/50" />
                        <span className="text-xs text-white/70">{image.metadata.taken_time}</span>
                      </div>
                    )}
                    {image.metadata?.camera && (
                      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10">
                        <Camera className="w-3.5 h-3.5 text-white/50" />
                        <span className="text-xs text-white/70">{image.metadata.camera}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Desktop Layout */}
            <div className="hidden lg:flex flex-row gap-8">
              {/* Image Container - Desktop */}
              <div className="flex-1 flex items-center justify-center overflow-hidden rounded-3xl bg-white/5 border border-white/10 backdrop-blur-sm relative">
                {isImageLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                    <div className="flex items-center gap-3 text-white/60">
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span className="text-sm">Loading image...</span>
                    </div>
                  </div>
                )}
                <motion.img
                  initial={{ scale: 1.1, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.6, ease: [0.25, 0.4, 0.25, 1] }}
                  src={image.original_url || "/placeholder.svg"}
                  alt={image.metadata?.title || "Gallery image"}
                  className="max-w-full max-h-[80vh] object-contain"
                  onLoad={() => setIsImageLoading(false)}
                  onError={() => setIsImageLoading(false)}
                />
              </div>

              {/* Details Panel - Desktop */}
              <motion.div
                initial={{ x: 40, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.2, ease: [0.25, 0.4, 0.25, 1] }}
                className="w-[420px] bg-white/5 backdrop-blur-2xl rounded-3xl p-8 border border-white/10 overflow-y-auto 
                         flex flex-col gap-6"
              >
                {/* Title Section */}
                {image.metadata?.title && (
                  <div className="space-y-2">
                    <h2 className="text-3xl font-bold text-white leading-tight">{image.metadata.title}</h2>
                    {showMatch && matchPercent !== null && (
                      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/10 border border-white/20">
                        <div className="w-2 h-2 rounded-full bg-white/60 animate-pulse" />
                        <span className="text-sm font-medium text-white/80">{matchPercent}% match</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Description */}
                {image.metadata?.description && (
                  <div className="space-y-3">
                    <h3 className="text-xs font-semibold text-white/40 uppercase tracking-wider">About</h3>
                    <p className="text-white/80 leading-relaxed text-base">{image.metadata.description}</p>
                  </div>
                )}

                {/* Metadata Grid */}
                <div className="space-y-4 pt-6 border-t border-white/10">
                  <h3 className="text-xs font-semibold text-white/40 uppercase tracking-wider">Details</h3>

                  <div className="grid gap-4">
                    {image.metadata?.taken_time && (
                      <div className="flex items-start gap-4 p-4 rounded-xl bg-white/5 border border-white/10">
                        <div className="p-2 rounded-lg bg-white/5">
                          <Calendar className="w-5 h-5 text-white/60" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-white/40 mb-1">Date Taken</p>
                          <p className="text-sm text-white/90 font-medium">{image.metadata.taken_time}</p>
                        </div>
                      </div>
                    )}

                    {image.metadata?.camera && (
                      <div className="flex items-start gap-4 p-4 rounded-xl bg-white/5 border border-white/10">
                        <div className="p-2 rounded-lg bg-white/5">
                          <Camera className="w-5 h-5 text-white/60" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-white/40 mb-1">Camera</p>
                          <p className="text-sm text-white/90 font-medium">{image.metadata.camera}</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col gap-3 pt-6 border-t border-white/10">
                  <button
                    onClick={() => window.open(image.original_url, "_blank")}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-white/10 hover:bg-white/20 
                             border border-white/10 hover:border-white/20 text-white font-medium transition-all duration-300"
                  >
                    <Fullscreen className="w-4 h-4" />
                    <span>View Full-Res</span>
                  </button>
                  <button
                    onClick={() => image && onSimilarTo?.(image)}
                    disabled={!onSimilarTo || isSimilarLoading}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-purple-500/20 hover:bg-purple-500/30 
                             border border-purple-500/30 hover:border-purple-500/40 text-white font-medium transition-all duration-300
                             disabled:opacity-60 disabled:cursor-not-allowed"
                  >
                    {isSimilarLoading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Finding similar...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" />
                        <span>Similar to this image</span>
                      </>
                    )}
                  </button>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
