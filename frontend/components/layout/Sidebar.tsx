"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { Network, Upload, MessageSquare, LogOut } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { useAuthStore } from "@/store/authStore"

const NAV_ITEMS = [
  {
    label: "Bulk Upload",
    href: "/bulk-upload",
    icon: Upload,
  },
  {
    label: "Chat",
    href: "/chat",
    icon: MessageSquare,
  },
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
    <aside className="flex h-full w-60 flex-col border-r border-border bg-card">
      {/* Logo */}
      <div className="flex h-16 shrink-0 items-center gap-2 border-b border-border px-5">
        <div className="flex size-8 items-center justify-center rounded-lg bg-primary">
          <Network className="size-4 text-primary-foreground" />
        </div>
        <span className="text-base font-semibold tracking-tight">RKG</span>
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col gap-1 overflow-y-auto p-3">
        <p className="mb-1 px-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Workspace
        </p>
        {NAV_ITEMS.map(({ label, href, icon: Icon }) => {
          const isActive = pathname === href || pathname.startsWith(href + "/")
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
              aria-current={isActive ? "page" : undefined}
            >
              <Icon
                className={cn(
                  "size-4 shrink-0",
                  isActive ? "text-primary" : "text-muted-foreground"
                )}
              />
              {label}
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="shrink-0 p-3">
        <Separator className="mb-3" />
        <Button
          variant="ghost"
          className="w-full justify-start gap-3 text-muted-foreground hover:text-destructive"
          onClick={handleLogout}
        >
          <LogOut className="size-4 shrink-0" />
          Sign Out
        </Button>
      </div>
    </aside>
  )
}
