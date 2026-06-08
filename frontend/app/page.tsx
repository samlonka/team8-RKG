import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Network,
  Brain,
  Upload,
  MessageSquare,
  Zap,
  Shield,
  ArrowRight,
} from "lucide-react"

const features = [
  {
    icon: Brain,
    title: "AI-Powered Reasoning",
    description:
      "Advanced reflexive knowledge graph that learns and adapts from your data, delivering intelligent insights at scale.",
  },
  {
    icon: Upload,
    title: "Bulk Data Ingestion",
    description:
      "Effortlessly upload and process large datasets. Vendor manifests, structured data, and unstructured documents.",
  },
  {
    icon: MessageSquare,
    title: "Conversational Analytics",
    description:
      "Ask questions in natural language and get precise, graph-grounded answers from your knowledge base.",
  },
  {
    icon: Network,
    title: "Knowledge Graph Visualization",
    description:
      "Explore entity relationships through interactive graph visualizations with drill-down capabilities.",
  },
  {
    icon: Zap,
    title: "Real-Time Reflection",
    description:
      "The graph reflects on its own structure, identifying gaps, contradictions, and hidden patterns automatically.",
  },
  {
    icon: Shield,
    title: "Enterprise-Grade Security",
    description:
      "Role-based access control, audit logs, and data isolation to meet compliance and governance requirements.",
  },
]

export default function LandingPage() {
  return (
    <div className="flex min-h-screen flex-col">
      {/* Navigation */}
      <header className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-2">
            <div className="flex size-8 items-center justify-center rounded-lg bg-primary">
              <Network className="size-4 text-primary-foreground" />
            </div>
            <span className="text-lg font-semibold tracking-tight">RKG</span>
          </div>
          <nav className="hidden items-center gap-6 md:flex">
            <a
              href="#features"
              className="text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              Features
            </a>
            <a
              href="#about"
              className="text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              About
            </a>
          </nav>
          <Link href="/login">
            <Button variant="default" size="sm">
              Sign In
              <ArrowRight />
            </Button>
          </Link>
        </div>
      </header>

      {/* Hero */}
      <main className="flex-1">
        <section className="relative overflow-hidden px-6 py-24 md:py-32">
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 -z-10"
          >
            <div className="absolute left-1/2 top-0 h-[600px] w-[900px] -translate-x-1/2 rounded-full bg-primary/5 blur-3xl" />
          </div>

          <div className="mx-auto max-w-4xl text-center">
            <Badge variant="secondary" className="mb-6">
              Hackathon 2025 · Team 8
            </Badge>
            <h1 className="mb-6 text-5xl font-bold tracking-tight md:text-6xl lg:text-7xl">
              Reflexive{" "}
              <span className="text-primary">Knowledge</span>
              <br />
              Graph Platform
            </h1>
            <p className="mx-auto mb-10 max-w-2xl text-lg text-muted-foreground md:text-xl">
              Transform raw data into structured intelligence. Our AI-powered
              platform builds, maintains, and queries knowledge graphs that
              reason about themselves.
            </p>
            <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
              <Link href="/login">
                <Button size="lg" className="gap-2">
                  Get Started
                  <ArrowRight className="size-4" />
                </Button>
              </Link>
              <a href="#features">
                <Button variant="outline" size="lg">
                  Explore Features
                </Button>
              </a>
            </div>
          </div>
        </section>

        {/* Stats */}
        <section className="border-y border-border/50 bg-muted/30 px-6 py-12">
          <div className="mx-auto grid max-w-4xl grid-cols-2 gap-8 md:grid-cols-4">
            {[
              { value: "10M+", label: "Entities Processed" },
              { value: "99.9%", label: "Uptime SLA" },
              { value: "<100ms", label: "Query Latency" },
              { value: "50+", label: "Data Connectors" },
            ].map(({ value, label }) => (
              <div key={label} className="text-center">
                <div className="text-3xl font-bold text-primary">{value}</div>
                <div className="mt-1 text-sm text-muted-foreground">{label}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Features */}
        <section id="features" className="px-6 py-24">
          <div className="mx-auto max-w-7xl">
            <div className="mb-16 text-center">
              <h2 className="mb-4 text-3xl font-bold tracking-tight md:text-4xl">
                Everything you need to work with knowledge
              </h2>
              <p className="mx-auto max-w-2xl text-muted-foreground">
                A complete platform for building, maintaining, and querying
                intelligent knowledge graphs at enterprise scale.
              </p>
            </div>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {features.map(({ icon: Icon, title, description }) => (
                <Card key={title}>
                  <CardHeader>
                    <div className="mb-2 flex size-10 items-center justify-center rounded-lg bg-primary/10">
                      <Icon className="size-5 text-primary" />
                    </div>
                    <CardTitle>{title}</CardTitle>
                    <CardDescription>{description}</CardDescription>
                  </CardHeader>
                </Card>
              ))}
            </div>
          </div>
        </section>

        {/* CTA */}
        <section
          id="about"
          className="bg-primary px-6 py-24 text-primary-foreground"
        >
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="mb-4 text-3xl font-bold tracking-tight md:text-4xl">
              Ready to unlock your data&apos;s full potential?
            </h2>
            <p className="mb-10 text-primary-foreground/80">
              Join the teams already using RKG to turn complex data into
              actionable intelligence.
            </p>
            <Link href="/login">
              <Button variant="secondary" size="lg" className="gap-2">
                Start Now
                <ArrowRight className="size-4" />
              </Button>
            </Link>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 px-6 py-8">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 sm:flex-row">
          <div className="flex items-center gap-2">
            <div className="flex size-6 items-center justify-center rounded bg-primary">
              <Network className="size-3 text-primary-foreground" />
            </div>
            <span className="text-sm font-medium">RKG</span>
          </div>
          <p className="text-sm text-muted-foreground">
            © 2025 Team 8 · Reflexive Knowledge Graph
          </p>
        </div>
      </footer>
    </div>
  )
}
