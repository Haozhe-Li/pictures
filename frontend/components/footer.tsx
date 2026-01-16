"use client"

export function Footer() {
  return (
    <footer className="relative border-t border-white/5 bg-black/40 backdrop-blur-xl mt-24">
      <div className="container mx-auto px-4 py-6">
        <p className="text-center text-sm text-white/40">
          © {new Date().getFullYear()} Haozhe Li. All rights reserved.
        </p>
      </div>
    </footer>
  )
}
