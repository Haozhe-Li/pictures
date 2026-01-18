"use client"

import type React from "react"
import { useState } from "react"
import { Search, Shuffle, X } from "lucide-react"
import { cn } from "@/lib/utils"

interface SearchBarProps {
  value: string
  onChange: (value: string) => void
  onSearch: (query: string) => void
  onClear: () => void
  isLoading?: boolean
  className?: string
}

export function SearchBar({ value, onChange, onSearch, onClear, isLoading, className }: SearchBarProps) {
  const [isLoadingRandom, setIsLoadingRandom] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (value.trim()) {
      onSearch(value.trim())
    }
  }

  const handleRandomQuery = async () => {
    setIsLoadingRandom(true)
    try {
      const res = await fetch("/api/random-query")
      const data = await res.json()
      if (data.query) {
        onChange(data.query)
        onSearch(data.query)
      }
    } catch (error) {
      console.error("Failed to get random query:", error)
    } finally {
      setIsLoadingRandom(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className={cn("relative", className)}>
      <div className="flex items-center gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-0 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground/60" />
          <input
            type="text"
            placeholder="Search..."
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="w-full pl-6 pr-8 py-2 bg-transparent border-b border-border/60 text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:border-foreground/30 transition-colors"
          />
          {value && (
            <button
              type="button"
              onClick={onClear}
              className="absolute right-0 top-1/2 -translate-y-1/2 p-1 text-muted-foreground/60 hover:text-foreground transition-colors"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <button
          type="button"
          onClick={handleRandomQuery}
          disabled={isLoadingRandom}
          className="p-1.5 text-muted-foreground/60 hover:text-foreground transition-colors disabled:opacity-50"
          title="Random search"
        >
          <Shuffle className={cn("h-4 w-4", isLoadingRandom && "animate-spin")} />
        </button>
      </div>
    </form>
  )
}
