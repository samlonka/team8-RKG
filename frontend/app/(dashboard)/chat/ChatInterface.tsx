"use client"

import { useState, useRef, useEffect } from "react"
import { Send, Bot, User2, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useAuthStore } from "@/store/authStore"
import {
  formatQueryResponse,
  queryAskStream,
  type PipelineEvent,
  type QueryAskResponse,
} from "@/lib/rkg-api"
import {
  PipelineStatusLine,
  PipelineStepsCollapsible,
} from "@/components/chat/PipelineProgress"
import { QueryResultPanel } from "@/components/chat/QueryResultPanel"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  reasoning?: string
  pipelineEvents?: PipelineEvent[]
  queryResponse?: QueryAskResponse
  duplicateReport?: QueryAskResponse["duplicate_report"]
}

const SAMPLE_QUESTIONS = [
  "Is the product available in the master list: AQUA WATER 28OZ PL 1/15",
  "Why are so many brands created as duplicates during customer import?",
  "Rank all GlobalSKUs by risk of causing training failures.",
  "Which SKUs are shared across multiple customers and unsafe to change?",
]

function formatTime(date: Date): string {
  const hours = date.getHours().toString().padStart(2, "0")
  const minutes = date.getMinutes().toString().padStart(2, "0")
  return `${hours}:${minutes}`
}

function buildAssistantMessage(data: QueryAskResponse): Message {
  return {
    id: `assistant-${Date.now()}`,
    role: "assistant",
    content: formatQueryResponse(data),
    timestamp: new Date(),
    reasoning: data.best_reasoning,
    pipelineEvents: data.pipeline_events,
    queryResponse: data,
    duplicateReport: data.duplicate_report,
  }
}

export function ChatInterface() {
  const user = useAuthStore((s) => s.user)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: `Hi${user?.name ? ` ${user.name}` : ""}! Ask about master catalog lookups (e.g. "Is AQUA WATER 28OZ PL 1/15 in the master list?") or graph investigations (risk rank, import failures, shared SKUs).`,
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState("")
  const [isThinking, setIsThinking] = useState(false)
  const [livePipeline, setLivePipeline] = useState<PipelineEvent[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isThinking, livePipeline])

  async function sendMessage(content: string) {
    const trimmed = content.trim()
    if (!trimmed || isThinking) return

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMsg])
    setInput("")
    setIsThinking(true)
    setLivePipeline([])

    try {
      const data = await queryAskStream(trimmed, (event) => {
        setLivePipeline((prev) => [...prev, event])
      })
      setMessages((prev) => [...prev, buildAssistantMessage(data)])
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Request failed"
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: `Could not reach the RKG API (${msg}). Start the backend with \`uvicorn api.main:app --reload --port 8000\`.`,
          timestamp: new Date(),
        },
      ])
    } finally {
      setIsThinking(false)
      setLivePipeline([])
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  return (
    <div className="flex h-full flex-col gap-0 overflow-hidden rounded-xl border border-border bg-card">
      <div className="flex items-center gap-3 border-b border-border px-5 py-4">
        <div className="flex size-8 items-center justify-center rounded-lg bg-primary/10">
          <Sparkles className="size-4 text-primary" />
        </div>
        <div>
          <p className="text-sm font-medium">Knowledge Graph Assistant</p>
          <p className="text-xs text-muted-foreground">
            Catalog lookups & graph investigations
          </p>
        </div>
        <div className="ml-auto flex size-2 rounded-full bg-emerald-500" />
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-6">
        <div className="flex flex-col gap-6">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={cn(
                "flex items-start gap-3",
                msg.role === "user" && "flex-row-reverse"
              )}
            >
              <div
                className={cn(
                  "flex size-8 shrink-0 items-center justify-center rounded-lg",
                  msg.role === "assistant" ? "bg-primary/10" : "bg-muted"
                )}
              >
                {msg.role === "assistant" ? (
                  <Bot className="size-4 text-primary" />
                ) : (
                  <User2 className="size-4 text-muted-foreground" />
                )}
              </div>

              <div
                className={cn(
                  "max-w-[min(100%,52rem)] rounded-xl px-4 py-3 text-sm",
                  msg.role === "assistant"
                    ? "bg-muted text-foreground"
                    : "max-w-[75%] bg-primary text-primary-foreground"
                )}
              >
                <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                {msg.queryResponse ? (
                  <QueryResultPanel data={msg.queryResponse} />
                ) : null}
                {msg.pipelineEvents && msg.pipelineEvents.length > 0 ? (
                  <PipelineStepsCollapsible events={msg.pipelineEvents} />
                ) : null}
                <p
                  className={cn(
                    "mt-1.5 text-right text-[10px]",
                    msg.role === "assistant"
                      ? "text-muted-foreground"
                      : "text-primary-foreground/70"
                  )}
                >
                  {formatTime(msg.timestamp)}
                </p>
              </div>
            </div>
          ))}

          {isThinking && (
            <div className="flex items-start gap-3">
              <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                <Bot className="size-4 text-primary" />
              </div>
              <div className="max-w-[75%] rounded-xl bg-muted px-4 py-3">
                <PipelineStatusLine events={livePipeline} live />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {messages.length === 1 && (
        <div className="border-t border-border px-5 py-3">
          <p className="mb-2 text-xs font-medium text-muted-foreground">
            Try asking:
          </p>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_QUESTIONS.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => sendMessage(q)}
                className="rounded-full border border-border bg-background px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground"
              >
                {q.length > 48 ? `${q.slice(0, 48)}…` : q}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="border-t border-border p-4">
        <div className="flex items-end gap-3 rounded-xl border border-border bg-background px-4 py-3 focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50">
          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about master catalog or the knowledge graph…"
            disabled={isThinking}
            className="flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
            style={{ minHeight: "24px", maxHeight: "120px" }}
          />
          <Button
            size="icon-sm"
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || isThinking}
            aria-label="Send message"
          >
            <Send className="size-3.5" />
          </Button>
        </div>
      </div>
    </div>
  )
}
