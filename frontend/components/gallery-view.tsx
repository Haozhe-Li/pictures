"use client"

import type React from "react"
import { Footer } from "@/components/footer"
import { useState, useEffect, useCallback } from "react"
import { Search, X, Loader2, Sparkles, Dices } from "lucide-react"
import { UploadModal } from "@/components/upload-modal"
import { ImageGallery } from "@/components/image-gallery"
import { ImageModal } from "@/components/image-modal"
import type { GalleryImage } from "@/types/gallery"
import { motion } from "framer-motion"

interface GalleryViewProps {
    initialImages?: GalleryImage[]
    initialNextCursor?: string | null
    initialLoaded?: boolean
}

export function GalleryView({ initialImages = [], initialNextCursor = null, initialLoaded = false }: GalleryViewProps) {
    const [images, setImages] = useState<GalleryImage[]>(initialImages)
    const [loading, setLoading] = useState(!initialLoaded && initialImages.length === 0)
    const [searchQuery, setSearchQuery] = useState("")
    const [isSearching, setIsSearching] = useState(false)
    const [isSimilarSearching, setIsSimilarSearching] = useState(false)
    const [isRandomLoading, setIsRandomLoading] = useState(false)
    const [isSearchResultView, setIsSearchResultView] = useState(false)
    const [selectedImage, setSelectedImage] = useState<GalleryImage | null>(null)
    const [nextCursor, setNextCursor] = useState<string | null>(initialNextCursor)
    const [lastFetchWasLoadMore, setLastFetchWasLoadMore] = useState(false)
    const [isUploadOpen, setIsUploadOpen] = useState(false)
    const [isDraggingOver, setIsDraggingOver] = useState(false)
    const [droppedFiles, setDroppedFiles] = useState<File[]>([])

    // Global drag-drop handler for upload
    useEffect(() => {
        let dragCounter = 0

        const handleDragEnter = (e: DragEvent) => {
            e.preventDefault()
            e.stopPropagation()
            dragCounter++
            if (e.dataTransfer?.types.includes('Files')) {
                setIsDraggingOver(true)
            }
        }

        const handleDragLeave = (e: DragEvent) => {
            e.preventDefault()
            e.stopPropagation()
            dragCounter--
            if (dragCounter === 0) {
                setIsDraggingOver(false)
            }
        }

        const handleDragOver = (e: DragEvent) => {
            e.preventDefault()
            e.stopPropagation()
        }

        const handleDrop = (e: DragEvent) => {
            e.preventDefault()
            e.stopPropagation()
            dragCounter = 0
            setIsDraggingOver(false)

            const files = e.dataTransfer?.files
            if (files && files.length > 0) {
                const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/'))
                if (imageFiles.length > 0) {
                    setDroppedFiles(imageFiles)
                    setIsUploadOpen(true)
                }
            }
        }

        document.addEventListener('dragenter', handleDragEnter)
        document.addEventListener('dragleave', handleDragLeave)
        document.addEventListener('dragover', handleDragOver)
        document.addEventListener('drop', handleDrop)

        return () => {
            document.removeEventListener('dragenter', handleDragEnter)
            document.removeEventListener('dragleave', handleDragLeave)
            document.removeEventListener('dragover', handleDragOver)
            document.removeEventListener('drop', handleDrop)
        }
    }, [])

    // Clear dropped files when modal closes
    const handleUploadClose = useCallback(() => {
        setIsUploadOpen(false)
        setDroppedFiles([])
    }, [])

    // Only fetch on mount if we don't have initial images
    useEffect(() => {
        if (!initialLoaded && initialImages.length === 0) {
            fetchGallery()
        } else {
            setLoading(false)
        }
    }, []) // eslint-disable-line react-hooks/exhaustive-deps

    const fetchGallery = async (cursor?: string) => {
        try {
            setLoading(true)
            setLastFetchWasLoadMore(Boolean(cursor))
            const url = new URL("/api/gallery", window.location.origin)
            url.searchParams.append("limit", "20")
            if (cursor) {
                url.searchParams.append("cursor", cursor)
            }

            const response = await fetch(url.toString())
            const data = await response.json()

            if (data.error) {
                console.error("Gallery error:", data.error)
                setImages((prev) => (cursor ? prev : []))
                // setNextCursor(null) // Keep existing cursor on error if appending? Or reset. Safe to keep prev state if error.
                return
            }

            const items = Array.isArray(data.items) ? data.items : []
            setImages((prev) => (cursor ? [...prev, ...items] : items))
            setNextCursor(data.next_cursor || null)
        } catch (error) {
            console.error("Failed to fetch gallery:", error)
            // setImages([]) // Don't wipe potentially existing images on fetch error
        } finally {
            setLoading(false)
        }
    }

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault()

        if (!searchQuery.trim()) {
            // Reset to initial state or re-fetch active non-search
            setImages(initialImages.length > 0 ? initialImages : [])
            setNextCursor(initialNextCursor)
            setIsSearchResultView(false)
            // Or maybe fetch fresh gallery
            fetchGallery()
            return
        }

        try {
            setIsSearching(true)
            const response = await fetch("/api/search", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    query: searchQuery,
                    limit: 20,
                }),
            })

            const data = await response.json()
            setImages(Array.isArray(data) ? data : [])
            setNextCursor(null) // No pagination for search results
            setIsSearchResultView(true)
            // Scroll to top after search completes
            window.scrollTo({ top: 0, behavior: 'smooth' })
        } catch (error) {
            console.error("Search failed:", error)
            setImages([])
            setIsSearchResultView(false)
        } finally {
            setIsSearching(false)
        }
    }

    const handleRandomQuery = async () => {
        try {
            setIsRandomLoading(true)
            const response = await fetch("/api/generate-random-query")
            const data = await response.json()

            if (data.query) {
                setSearchQuery(data.query)

                // Execute search with the random query
                setIsSearching(true)
                const searchResponse = await fetch("/api/search", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        query: data.query,
                        limit: 20,
                    }),
                })

                const searchData = await searchResponse.json()
                setImages(Array.isArray(searchData) ? searchData : [])
                setNextCursor(null)
                setIsSearchResultView(true)
                window.scrollTo({ top: 0, behavior: 'smooth' })
            }
        } catch (error) {
            console.error("Random query failed:", error)
        } finally {
            setIsRandomLoading(false)
            setIsSearching(false)
        }
    }

    const handleSimilarTo = async (image: GalleryImage) => {
        if (!image?.original_url) {
            return
        }

        try {
            setIsSimilarSearching(true)
            const title = image.metadata?.title || "this image"
            setSearchQuery(`Images similar to ${title}`)

            const response = await fetch("/api/similar-to", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    image_url: image.original_url,
                    limit: 20,
                }),
            })

            const data = await response.json()
            setImages(Array.isArray(data) ? data : [])
            setNextCursor(null)
            setSelectedImage(null)
            setIsSearchResultView(true)
            // Scroll to top after search completes
            window.scrollTo({ top: 0, behavior: 'smooth' })
        } catch (error) {
            console.error("Similar-to search failed:", error)
            setImages([])
            setIsSearchResultView(false)
        } finally {
            setIsSimilarSearching(false)
        }
    }

    const clearSearch = () => {
        setSearchQuery("")
        if (initialImages.length > 0) {
            setImages(initialImages)
            setNextCursor(initialNextCursor)
        } else {
            fetchGallery()
        }
        setIsSearchResultView(false)
    }

    const showMatch = isSearching || isSimilarSearching || isSearchResultView

    return (
        <>
            {/* Drag overlay */}
            {isDraggingOver && (
                <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center pointer-events-none">
                    <div className="text-center space-y-4">
                        <div className="w-24 h-24 mx-auto rounded-full bg-purple-500/20 border-2 border-dashed border-purple-500/50 flex items-center justify-center">
                            <Sparkles className="w-10 h-10 text-purple-400" />
                        </div>
                        <p className="text-xl text-white font-medium">Drop images to upload</p>
                        <p className="text-sm text-white/50">Release to open upload dialog</p>
                    </div>
                </div>
            )}

            <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-black to-zinc-900">
                <div className="relative">
                    {/* Ambient gradient effects */}
                    <div className="absolute inset-0 overflow-hidden pointer-events-none">
                        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl" />
                        <div className="absolute top-20 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
                    </div>

                    <div className="relative">
                        {/* Hero Header */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8 }}
                            className="container mx-auto px-4 pt-20 pb-12"
                        >
                            <div className="max-w-4xl mx-auto text-center space-y-8">
                                {/* Title */}
                                <div className="space-y-4">
                                    <motion.div
                                        initial={{ scale: 0.9, opacity: 0 }}
                                        animate={{ scale: 1, opacity: 1 }}
                                        transition={{ duration: 0.6, delay: 0.2 }}
                                        className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-sm"
                                    >
                                        <Sparkles className="w-4 h-4 text-purple-400" />
                                        <span className="text-sm text-white/70 font-medium">AI-Powered Gallery</span>
                                    </motion.div>

                                    <h1 className="text-4xl sm:text-5xl md:text-7xl font-bold tracking-tight">
                                        <a
                                            href="/"
                                            onClick={(e) => { e.preventDefault(); window.location.reload(); }}
                                            className="bg-gradient-to-r from-white via-white to-white/60 bg-clip-text text-transparent hover:from-white hover:via-white hover:to-white/80 transition-all cursor-pointer"
                                        >
                                            Visual Collection
                                        </a>
                                    </h1>

                                    <p className="text-lg text-white/50 max-w-2xl mx-auto leading-relaxed">
                                        Explore my works with help of AI
                                    </p>
                                </div>

                                {/* Search Bar */}
                                <motion.form
                                    onSubmit={handleSearch}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.6, delay: 0.4 }}
                                    className="relative max-w-2xl mx-auto space-y-3"
                                >
                                    {/* Input Container */}
                                    <div className="relative group">
                                        {/* Search icon */}
                                        <div className="absolute left-4 top-1/2 -translate-y-1/2 z-10">
                                            <Search className="w-5 h-5 text-white/40 group-focus-within:text-white/60 transition-colors" />
                                        </div>

                                        {/* Input */}
                                        <input
                                            type="text"
                                            placeholder="Describe what you're looking for..."
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            className="w-full h-14 pl-12 pr-14 rounded-2xl bg-white/5 border border-white/10 text-white placeholder:text-white/30 
                                               focus:outline-none focus:ring-2 focus:ring-white/20 focus:border-white/20 focus:bg-white/10
                                               backdrop-blur-xl transition-all duration-300 text-base"
                                        />

                                        {/* Right side button - either dice or clear */}
                                        <div className="absolute right-3 top-1/2 -translate-y-1/2">
                                            {searchQuery ? (
                                                <motion.button
                                                    initial={{ scale: 0, opacity: 0 }}
                                                    animate={{ scale: 1, opacity: 1 }}
                                                    exit={{ scale: 0, opacity: 0 }}
                                                    type="button"
                                                    onClick={clearSearch}
                                                    className="group/btn flex items-center gap-0 sm:hover:gap-2 p-2 sm:hover:pl-3 sm:hover:pr-4 rounded-full bg-white/10 hover:bg-white/20 transition-all duration-300"
                                                >
                                                    <X className="w-4 h-4 text-white/60 shrink-0" />
                                                    <span className="max-w-0 sm:group-hover/btn:max-w-[60px] overflow-hidden whitespace-nowrap text-sm text-white/70 transition-all duration-300">
                                                        Clear
                                                    </span>
                                                </motion.button>
                                            ) : (
                                                <button
                                                    type="button"
                                                    onClick={handleRandomQuery}
                                                    disabled={isSearching || isSimilarSearching || isRandomLoading}
                                                    className="group/btn flex items-center gap-0 sm:hover:gap-2 p-2 sm:hover:pl-3 sm:hover:pr-4 rounded-full bg-white/10 hover:bg-white/20 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                                                    title="Surprise me"
                                                >
                                                    {isRandomLoading ? (
                                                        <Loader2 className="w-4 h-4 text-white/60 animate-spin shrink-0" />
                                                    ) : (
                                                        <Dices className="w-4 h-4 text-white/60 shrink-0" />
                                                    )}
                                                    <span className="max-w-0 sm:group-hover/btn:max-w-[80px] overflow-hidden whitespace-nowrap text-sm text-white/70 transition-all duration-300">
                                                        {isRandomLoading ? "Loading" : "Surprise"}
                                                    </span>
                                                </button>
                                            )}
                                        </div>

                                        {/* Glow effect on focus */}
                                        <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/20 via-blue-500/20 to-purple-500/20 opacity-0 group-focus-within:opacity-100 blur-xl transition-opacity duration-500 -z-10" />
                                    </div>

                                    {/* Search hint */}
                                    <motion.p
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        transition={{ delay: 0.8 }}
                                        className="text-xs text-white/30 text-center"
                                    >
                                        Press Enter to search • Click dice for a surprise
                                    </motion.p>
                                </motion.form>
                            </div>
                        </motion.div>
                    </div>
                </div>

                {/* Main Content */}
                <main className="container mx-auto px-4 py-12">
                    {loading && images.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-96 gap-4">
                            <Loader2 className="w-10 h-10 animate-spin text-white/30" />
                            <p className="text-white/40 text-sm">Loading gallery...</p>
                        </div>
                    ) : (
                        <>
                            <ImageGallery
                                images={images}
                                onImageClick={setSelectedImage}
                                isSearching={isSearching || isSimilarSearching}
                                staggerDelay={lastFetchWasLoadMore ? 0 : 0.05}
                                showMatch={showMatch}
                            />

                            {nextCursor && !searchQuery && (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="flex justify-center mt-16"
                                >
                                    <button
                                        onClick={() => fetchGallery(nextCursor)}
                                        disabled={loading}
                                        className="group relative px-8 py-4 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 
                               text-white font-medium transition-all duration-300 overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        <span className="relative z-10 flex items-center gap-2">
                                            {loading ? (
                                                <>
                                                    <Loader2 className="w-4 h-4 animate-spin" />
                                                    Loading...
                                                </>
                                            ) : (
                                                "Load More"
                                            )}
                                        </span>
                                        <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                                    </button>
                                </motion.div>
                            )}
                        </>
                    )}
                </main>

                {/* Image Modal */}
                <ImageModal
                    image={selectedImage}
                    onClose={() => setSelectedImage(null)}
                    onSimilarTo={handleSimilarTo}
                    isSimilarLoading={isSimilarSearching}
                    showMatch={showMatch}
                />

                {/* Upload Modal */}
                <UploadModal
                    isOpen={isUploadOpen}
                    onClose={handleUploadClose}
                    initialFiles={droppedFiles}
                />
            </div>
            <Footer />
        </>
    )
}
