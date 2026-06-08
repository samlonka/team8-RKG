import type { Metadata } from "next"
import { ChatInterface } from "./ChatInterface"

export const metadata: Metadata = {
  title: "Chat",
}

export default function ChatPage() {
  return (
    <div className="mx-auto flex h-full max-w-4xl flex-col">
      <ChatInterface />
    </div>
  )
}
