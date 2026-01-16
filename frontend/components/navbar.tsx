"use client"

import { motion } from "framer-motion"
import { Info, Upload } from "lucide-react"
import Link from "next/link"
import Image from "next/image"
import { useState } from "react"
import { UploadModal } from "./upload-modal"

export function Navbar() {
  const [isUploadOpen, setIsUploadOpen] = useState(false)

  return (
    <>
      <motion.nav
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 backdrop-blur-xl bg-black/40"
      >
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center gap-2 group" onClick={() => window.location.reload()}>
              <div className="w-9 h-9 rounded-lg bg-white/5 group-hover:bg-white/10 transition-colors overflow-hidden">
                <Image
                  src="/android-chrome-192x192.png"
                  alt="Site logo"
                  width={36}
                  height={36}
                  className="w-full h-full object-cover"
                />
              </div>
              {/* <span className="text-lg font-semibold text-white">Haozheli.Pictures</span> */}
            </Link>

            <div className="flex items-center gap-6">
              <button
                onClick={() => setIsUploadOpen(true)}
                className="flex items-center gap-2 text-sm text-white/60 hover:text-white transition-colors"
              >
                <Upload className="w-4 h-4" />
                <span>Upload</span>
              </button>
            </div>
          </div>
        </div>
      </motion.nav>

      <UploadModal isOpen={isUploadOpen} onClose={() => setIsUploadOpen(false)} />
    </>
  )
}
