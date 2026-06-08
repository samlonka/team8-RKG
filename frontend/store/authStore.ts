"use client"

import { create } from "zustand"
import { persist } from "zustand/middleware"
import type { User, LoginCredentials, AuthState } from "@/types/auth"

const AUTH_COOKIE = "rkg-auth-token"

function setAuthCookie(value: string) {
  const maxAge = 60 * 60 * 24 * 7 // 7 days
  document.cookie = `${AUTH_COOKIE}=${value}; path=/; max-age=${maxAge}; SameSite=Lax`
}

function clearAuthCookie() {
  document.cookie = `${AUTH_COOKIE}=; path=/; max-age=0`
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      login: ({ email }: LoginCredentials) => {
        const name = email.split("@")[0]
        const user: User = { email, name }
        setAuthCookie("authenticated")
        set({ user, isAuthenticated: true })
      },
      logout: () => {
        clearAuthCookie()
        set({ user: null, isAuthenticated: false })
      },
    }),
    {
      name: "rkg-auth-storage",
    }
  )
)
