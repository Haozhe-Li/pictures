"use client"

import { useCallback, useState, type DragEvent } from "react"
import { Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { UploadModal } from "@/components/upload-modal"
import { cn } from "@/lib/utils"

export function UploadTool() {
    const [isUploadOpen, setIsUploadOpen] = useState(false)
    const [droppedFiles, setDroppedFiles] = useState<File[]>([])
    const [isDragging, setIsDragging] = useState(false)

    const handleClose = useCallback(() => {
        setIsUploadOpen(false)
        setDroppedFiles([])
    }, [])

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault()
        e.dataTransfer.dropEffect = "copy"
    }

    const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault()
        if (e.dataTransfer?.types.includes("Files")) {
            setIsDragging(true)
        }
    }

    const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault()
        if (e.currentTarget.contains(e.relatedTarget as Node)) {
            return
        }
        setIsDragging(false)
    }

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault()
        setIsDragging(false)
        const files = e.dataTransfer?.files
        if (!files || files.length === 0) return

        const imageFiles = Array.from(files).filter((file) => file.type.startsWith("image/"))
        if (imageFiles.length > 0) {
            setDroppedFiles(imageFiles)
            setIsUploadOpen(true)
        }
    }

    return (
        <div className="min-h-screen bg-[#F2F2F2] text-gray-800">
            <main className="container mx-auto px-4 py-16">
                <div className="max-w-3xl mx-auto text-center space-y-4">
                    <h1 className="text-3xl sm:text-4xl font-normal tracking-tight font-serif text-gray-700">
                        Image Uploading Portal
                    </h1>
                    <p className="text-base text-gray-500">
                        Extracts EXIF automatically and supports batch uploads.
                    </p>
                </div>

                <div className="max-w-3xl mx-auto mt-10">
                    <div
                        className={cn(
                            "relative rounded-2xl border border-gray-300 bg-[#F7F7F7] p-10 text-center transition-colors",
                            isDragging ? "border-blue-400 bg-blue-50" : "hover:border-gray-400"
                        )}
                        onDragOver={handleDragOver}
                        onDragEnter={handleDragEnter}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                    >
                        <div className="mx-auto w-12 h-12 rounded-full bg-white border border-gray-200 flex items-center justify-center">
                            <Upload className="w-6 h-6 text-gray-400" />
                        </div>
                        <h2 className="mt-4 text-lg font-normal font-serif text-gray-700">Drop images here</h2>
                        <p className="mt-2 text-sm text-gray-500">Or click the button to open the uploader</p>

                        <div className="mt-6 flex items-center justify-center">
                            <Button
                                onClick={() => setIsUploadOpen(true)}
                                className="bg-gray-900 text-white hover:bg-gray-800"
                            >
                                Open uploader
                            </Button>
                        </div>

                        {isDragging && (
                            <div className="absolute inset-0 rounded-2xl border-2 border-dashed border-blue-400 pointer-events-none" />
                        )}
                    </div>

                    <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm text-gray-500">
                        <div className="rounded-xl border border-gray-300 bg-white p-4">
                            Supports JPG / PNG
                        </div>
                        <div className="rounded-xl border border-gray-300 bg-white p-4">
                            Auto EXIF extraction
                        </div>
                        <div className="rounded-xl border border-gray-300 bg-white p-4">
                            Batch upload with status
                        </div>
                    </div>
                </div>
            </main>

            <footer className="bg-[#F2F2F2]">
                <div className="container mx-auto px-4 py-6 text-center text-sm text-gray-500">
                    &copy;
                    <a
                        href="https://haozhe.li"
                        target="_blank"
                        rel="noreferrer"
                        className="font-normal hover:font-semibold hover:underline underline-offset-4 transition-all"
                    >
                        Haozhe Li
                    </a>
                    &nbsp;2026. All rights reserved.

                </div>
            </footer>

            <UploadModal isOpen={isUploadOpen} onClose={handleClose} initialFiles={droppedFiles} />
        </div>
    )
}
