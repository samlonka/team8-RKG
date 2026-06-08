"use client"

import { usePathname } from "next/navigation"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { useAuthStore } from "@/store/authStore"

const PAGE_TITLES: Record<string, { title: string; description: string }> = {
  "/bulk-upload": {
    title: "Bulk Upload",
    description: "Ingest datasets and documents into the knowledge graph",
  },
  "/chat": {
    title: "Chat",
    description: "Ask questions and explore your knowledge graph",
  },
}

function getInitials(name: string): string {
  return name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2)
}

export function DashboardHeader() {
  const pathname = usePathname()
  const user = useAuthStore((s) => s.user)

  const pageInfo = PAGE_TITLES[pathname] ?? {
    title: "Dashboard",
    description: "Manage your knowledge graph",
  }

  const initials = user ? getInitials(user.name) : "U"

  return (
    <header className="flex h-16 shrink-0 items-center gap-4 border-b border-border bg-background px-6">
      <div className="flex-1">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-semibold">{pageInfo.title}</h1>
          <Badge variant="secondary" className="hidden sm:inline-flex">
            Beta
          </Badge>
        </div>
        <p className="hidden text-xs text-muted-foreground sm:block">
          {pageInfo.description}
        </p>
      </div>

      <Separator orientation="vertical" className="h-6" />

      {/* User info */}
      <div className="flex items-center gap-2.5">
        <div className="hidden text-right sm:block">
          <p className="text-sm font-medium leading-none">{user?.name ?? "User"}</p>
          <p className="text-xs text-muted-foreground">{user?.email ?? ""}</p>
        </div>
        <Avatar className="size-8">
          <AvatarFallback className="bg-primary/10 text-xs font-medium text-primary">
            {initials}
          </AvatarFallback>
        </Avatar>
      </div>
    </header>
  )
}
