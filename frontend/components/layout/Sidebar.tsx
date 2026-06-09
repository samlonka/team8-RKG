"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { Network, Upload, MessageSquare, LogOut } from "lucide-react"
import { cn } from "@/lib/utils"
import { useAuthStore } from "@/store/authStore"

const NAV_ITEMS = [
  { label: "Bulk Upload", href: "/bulk-upload", icon: Upload },
  { label: "Chat", href: "/chat", icon: MessageSquare },
]

export function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const logout = useAuthStore((s) => s.logout)

  function handleLogout() {
    logout()
    router.push("/")
  }

  return (
    <aside className="flex h-full w-60 flex-col border-r border-sidebar-border bg-sidebar">
      {/* Logo */}
      <div className="flex h-16 shrink-0 items-center gap-2.5 border-b border-sidebar-border px-5">
        <div className="flex size-8 items-center justify-center rounded-lg bg-primary">
          <Network className="size-4 text-primary-foreground" />
        </div>
        <span className="text-[15px] font-semibold tracking-tight text-sidebar-foreground">
          RKG
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col gap-0.5 overflow-y-auto px-3 py-4">
        <p className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-widest text-sidebar-foreground/40">
          Workspace
        </p>
        {NAV_ITEMS.map(({ label, href, icon: Icon }) => {
          const isActive = pathname === href || pathname.startsWith(href + "/")
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-all duration-150",
                isActive
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              )}
              aria-current={isActive ? "page" : undefined}
            >
              <Icon
                className={cn(
                  "size-4 shrink-0",
                  isActive ? "text-primary-foreground" : "text-sidebar-foreground/50"
                )}
              />
              {label}
            </Link>
          )
        })}
      </nav>

      {/* Footer / logout */}
      <div className="shrink-0 border-t border-sidebar-border px-3 py-3">
        <button
          type="button"
          onClick={handleLogout}
          className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-sidebar-foreground/60 transition-all duration-150 hover:bg-sidebar-accent hover:text-destructive"
        >
          <LogOut className="size-4 shrink-0" />
          Sign Out
        </button>
      </div>
    </aside>
  )
}
