"use client"

import type React from "react"

import { motion, AnimatePresence } from "framer-motion"
import { X, Upload, Loader2, CheckCircle2, Trash2 } from "lucide-react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { extractExifData } from "@/lib/exif"
import { cn } from "@/lib/utils"

interface UploadItem {
  id: string
  file: File
  previewUrl: string
  title: string
  description: string
  takenTime: string
  camera: string
  status: "pending" | "uploading" | "success" | "error"
  error?: string
}

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
  initialFiles?: File[]
}

export function UploadModal({ isOpen, onClose, initialFiles }: UploadModalProps) {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [rememberPassword, setRememberPassword] = useState(false)

  const [items, setItems] = useState<UploadItem[]>([])
  const [currentItemId, setCurrentItemId] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const [isUploading, setIsUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const authStorageKey = "upload_auth_v1"

  // Handle initialFiles when they are provided (from drag-drop on page)
  useEffect(() => {
    if (isOpen && initialFiles && initialFiles.length > 0) {
      handleFiles(initialFiles)
    }
  }, [isOpen, initialFiles])

  useEffect(() => {
    try {
      const saved = localStorage.getItem(authStorageKey)
      if (saved) {
        const parsed = JSON.parse(saved) as { username?: string; password?: string }
        if (parsed.username) setUsername(parsed.username)
        if (parsed.password) setPassword(parsed.password)
        setRememberPassword(true)
      }
    } catch (err) {
      console.warn("Failed to load saved credentials", err)
    }
  }, [])

  useEffect(() => {
    if (!rememberPassword) {
      return
    }

    try {
      localStorage.setItem(authStorageKey, JSON.stringify({ username, password }))
    } catch (err) {
      console.warn("Failed to save credentials", err)
    }
  }, [rememberPassword, username, password])

  // 当 items 变化且没有 currentItemId 时，选中第一个
  useEffect(() => {
    if (items.length > 0 && !currentItemId) {
      setCurrentItemId(items[0].id)
    }
    if (items.length === 0) {
      setCurrentItemId(null)
    }
  }, [items, currentItemId])

  // Effect to clean up object URLs when modal closes or unmounts
  useEffect(() => {
    // Typically we might want to delay cleanup to not flash
    // But here we can rely on resetForm or similar manually called

    return () => {
      items.forEach(i => URL.revokeObjectURL(i.previewUrl))
    }
  }, [])

  const handleFiles = async (filesInput: FileList | File[]) => {
    const files = Array.from(filesInput).filter((file) => file.type.startsWith("image/"))
    if (files.length === 0) {
      return
    }

    const newItems: UploadItem[] = []

    for (const file of files) {
      const id = Math.random().toString(36).substring(7)
      const previewUrl = URL.createObjectURL(file)

      let takenTime = ""
      let camera = ""

      try {
        const exifData = await extractExifData(file)
        takenTime = exifData.takenTime || ""
        camera = exifData.camera || ""
      } catch (err) {
        console.warn("Failed to extract EXIF for", file.name, err)
      }

      newItems.push({
        id,
        file,
        previewUrl,
        title: "",
        description: "",
        takenTime,
        camera,
        status: "pending"
      })
    }

    setItems((prev) => [...prev, ...newItems])
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files
    if (selectedFiles && selectedFiles.length > 0) {
      await handleFiles(selectedFiles)
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = "copy"
  }

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    if (e.currentTarget.contains(e.relatedTarget as Node)) {
      return
    }
    setIsDragging(false)
  }

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await handleFiles(e.dataTransfer.files)
      e.dataTransfer.clearData()
    }
  }

  const updateItem = (id: string, updates: Partial<UploadItem>) => {
    setItems((prev) => prev.map((item) => (item.id === id ? { ...item, ...updates } : item)))
  }

  const removeItem = (id: string) => {
    setItems((prev) => {
      const newItems = prev.filter((item) => item.id !== id)
      return newItems
    })
    // If we removed the current item, logic in useEffect will handle picking a new one or setting to null
    if (currentItemId === id) {
      setCurrentItemId(null)
    }
  }

  const handleSubmit = async () => {
    if (items.length === 0 || !username || !password) return

    // Apply default titles if missing
    setItems(prev => prev.map(item => ({ ...item, title: item.title || item.file.name.split('.')[0] })))

    setIsUploading(true)

    // Filter pending items
    const pendingItems = items.filter(item => item.status !== 'success')
    const batchSize = 3

    for (let i = 0; i < pendingItems.length; i += batchSize) {
      const batch = pendingItems.slice(i, i + batchSize)

      // Delay between batches (except the first one)
      if (i > 0) {
        await new Promise(resolve => setTimeout(resolve, 5000))
      }

      // Process batch in parallel
      await Promise.all(batch.map(async (item) => {
        updateItem(item.id, { status: "uploading" })

        try {
          const formData = new FormData()
          formData.append("file", item.file)
          formData.append("title", item.title || item.file.name.split('.')[0])
          if (item.description) formData.append("description", item.description)
          if (item.takenTime) formData.append("taken_time", item.takenTime)
          if (item.camera) formData.append("camera", item.camera)

          const response = await fetch("/api/ingest", {
            method: "POST",
            headers: {
              Authorization: `Basic ${btoa(`${username}:${password}`)}`,
            },
            body: formData,
          })

          const data = await response.json()

          if (!response.ok) {
            throw new Error(data.error || "Upload failed")
          }

          updateItem(item.id, { status: "success" })
        } catch (err) {
          updateItem(item.id, {
            status: "error",
            error: err instanceof Error ? err.message : "Upload failed"
          })
        }
      }))
    }

    setIsUploading(false)

    // Check success status roughly
    setUploadSuccess(true)

    setTimeout(() => {
      onClose()
      window.location.reload()
    }, 2000)
  }

  // Get current item
  const currentItem = items.find((i) => i.id === currentItemId)

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-5xl h-[85vh] flex flex-col bg-zinc-900 border border-white/10 rounded-2xl shadow-2xl z-50 overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-white/10 bg-zinc-900/95 backdrop-blur-sm shrink-0">
              <h2 className="text-2xl font-semibold text-white">Upload Images</h2>
              <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                <X className="w-5 h-5 text-white/60" />
              </button>
            </div>

            <div className="flex flex-1 overflow-hidden">
              {/* Left Sidebar: Settings & Image List */}
              <div className="w-80 border-r border-white/10 flex flex-col bg-black/20">
                {/* Authentication */}
                <div className="p-4 space-y-3 border-b border-white/10 shrink-0">
                  <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider">Authentication</h3>
                  <div className="space-y-2">
                    <Input
                      placeholder="Username"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      className="bg-white/5 border-white/10 text-white text-sm h-8"
                    />
                    <Input
                      type="password"
                      placeholder="Password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="bg-white/5 border-white/10 text-white text-sm h-8"
                    />
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="remember-password"
                        checked={rememberPassword}
                        onCheckedChange={(checked) => {
                          const isChecked = checked === true
                          setRememberPassword(isChecked)
                          if (!isChecked) {
                            try {
                              localStorage.removeItem(authStorageKey)
                            } catch (err) {
                              console.warn("Failed to clear saved credentials", err)
                            }
                          }
                        }}
                        className="border-white/20"
                      />
                      <Label htmlFor="remember-password" className="text-xs text-white/50">
                        Remember password
                      </Label>
                    </div>
                  </div>
                </div>

                {/* Image List */}
                <div className="flex-1 overflow-hidden flex flex-col min-h-0">
                  <div className="flex items-center justify-between p-4 pb-2">
                    <h3 className="text-xs font-medium text-white/40 uppercase tracking-wider">Images ({items.length})</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 text-xs text-blue-400 hover:text-blue-300 hover:bg-blue-400/10 px-2"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      + Add
                    </Button>
                  </div>
                  <ScrollArea className="flex-1 h-full px-2">
                    <div className="space-y-1 p-2">
                      {items.map((item) => (
                        <div
                          key={item.id}
                          onClick={() => setCurrentItemId(item.id)}
                          className={cn(
                            "flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-colors group relative",
                            currentItemId === item.id
                              ? "bg-white/10"
                              : "hover:bg-white/5"
                          )}
                        >
                          <div className="w-12 h-12 rounded overflow-hidden bg-black shrink-0 border border-white/10">
                            <img src={item.previewUrl} className="w-full h-full object-cover" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className={cn("text-sm font-medium truncate", !item.title && "text-white/40 italic")}>
                              {item.title || item.file.name}
                            </p>
                            <div className="flex items-center gap-2 mt-1">
                              <span className={cn("text-xs",
                                item.status === 'success' ? "text-green-400" :
                                  item.status === 'error' ? "text-red-400" :
                                    item.status === 'uploading' ? "text-blue-400" : "text-white/40"
                              )}>
                                {item.status === 'success' && "Uploaded"}
                                {item.status === 'error' && "Failed"}
                                {item.status === 'uploading' && "Uploading..."}
                                {item.status === 'pending' && ((item.file.size / 1024 / 1024).toFixed(1) + " MB")}
                              </span>
                            </div>
                          </div>
                          {/* Remove Button (only show on hover or selected) */}
                          <button
                            onClick={(e) => { e.stopPropagation(); removeItem(item.id); }}
                            className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 text-white/40 transition-opacity absolute right-2 top-1/2 -translate-y-1/2"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      ))}
                      {items.length === 0 && (
                        <div className="text-center py-8 text-white/20 text-sm">
                          No images selected
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </div>
              </div>

              {/* Main Content: Edit Area or Upload Prompt */}
              <div
                className="flex-1 flex flex-col bg-zinc-900 overflow-y-auto relative"
                onDragOver={handleDragOver}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {isDragging && (
                  <div className="absolute inset-0 z-20 bg-white/5 border-2 border-dashed border-white/30 rounded-2xl flex items-center justify-center pointer-events-none">
                    <div className="flex flex-col items-center gap-3 text-white/80">
                      <div className="w-14 h-14 rounded-full bg-white/10 flex items-center justify-center">
                        <Upload className="w-7 h-7" />
                      </div>
                      <span className="text-sm font-medium">Drop images to upload</span>
                    </div>
                  </div>
                )}
                {currentItem ? (
                  <div className="p-8 max-w-3xl mx-auto w-full space-y-8">
                    {/* Image Preview Large */}
                    <div className="aspect-video bg-black/40 rounded-xl border border-white/10 overflow-hidden flex items-center justify-center relative group">
                      <img src={currentItem.previewUrl} className="max-h-full max-w-full object-contain" />
                      <div className="absolute top-4 right-4 bg-black/60 backdrop-blur px-3 py-1 rounded-full text-xs text-white/80 border border-white/10">
                        {currentItem.file.name}
                      </div>
                    </div>

                    {/* Edit Form */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <Label className="text-white/60">Title</Label>
                          <Input
                            value={currentItem.title}
                            onChange={(e) => updateItem(currentItem.id, { title: e.target.value })}
                            placeholder="Image title"
                            className="bg-black/40 border-white/10 text-white"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label className="text-white/60">Description</Label>
                          <Textarea
                            value={currentItem.description}
                            onChange={(e) => updateItem(currentItem.id, { description: e.target.value })}
                            placeholder="Add a description..."
                            className="bg-black/40 border-white/10 text-white min-h-[100px]"
                          />
                        </div>
                      </div>
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <Label className="text-white/60">Taken Time</Label>
                          <Input
                            value={currentItem.takenTime}
                            onChange={(e) => updateItem(currentItem.id, { takenTime: e.target.value })}
                            className="bg-black/40 border-white/10 text-white"
                            placeholder="YYYY-MM-DD HH:MM:SS"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label className="text-white/60">Camera Info</Label>
                          <Input
                            value={currentItem.camera}
                            onChange={(e) => updateItem(currentItem.id, { camera: e.target.value })}
                            className="bg-black/40 border-white/10 text-white"
                            placeholder="Camera model, lens, settings..."
                          />
                        </div>

                        {currentItem.status === 'error' && (
                          <div className="p-3 bg-red-500/10 border border-red-500/20 rounded text-red-400 text-sm">
                            Error: {currentItem.error}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center p-12 text-center space-y-4">
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      className="w-full max-w-sm aspect-square border-2 border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center gap-4 cursor-pointer hover:border-white/20 hover:bg-white/5 transition-all group"
                    >
                      <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center group-hover:scale-110 transition-transform">
                        <Upload className="w-8 h-8 text-white/40" />
                      </div>
                      <div className="space-y-1">
                        <p className="font-medium text-white/80">Click or drag to upload images</p>
                        <p className="text-sm text-white/40">Support JPG, PNG</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Footer Actions */}
            <div className="p-6 border-t border-white/10 bg-zinc-900/95 backdrop-blur-sm flex justify-between items-center shrink-0">
              <div className="text-sm text-white/40">
                {items.length > 0 ? (
                  <span>{items.filter(i => i.status === 'success').length} / {items.length} uploaded</span>
                ) : (
                  <span>Ready to upload</span>
                )}
              </div>

              <div className="flex items-center gap-3">
                <Button variant="ghost" onClick={onClose} className="hover:bg-white/10 text-white">
                  Cancel
                </Button>
                <Button
                  onClick={handleSubmit}
                  disabled={isUploading || items.length === 0 || !username || !password}
                  className="bg-white text-black hover:bg-white/90 min-w-[140px]"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Uploading...
                    </>
                  ) : uploadSuccess ? (
                    <>
                      <CheckCircle2 className="w-4 h-4 mr-2 text-green-600" />
                      All Done
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      {items.length > 0 ? `Upload ${items.length} Images` : 'Upload Images'}
                    </>
                  )}
                </Button>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/jpeg,image/png,image/jpg"
              onChange={handleFileChange}
              className="hidden"
            />
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
