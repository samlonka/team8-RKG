import type { Metadata } from "next"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CloudUpload, FileText, CheckCircle2, Clock } from "lucide-react"
import { BulkUploadForm } from "./BulkUploadForm"

export const metadata: Metadata = {
  title: "Bulk Upload",
}

const recentUploads = [
  {
    name: "vendor_manifest_q1.csv",
    size: "2.4 MB",
    status: "completed",
    records: 1243,
    time: "2 hours ago",
  },
  {
    name: "product_catalog_v2.json",
    size: "8.1 MB",
    status: "completed",
    records: 5821,
    time: "Yesterday",
  },
  {
    name: "supplier_data_2025.xlsx",
    size: "1.2 MB",
    status: "processing",
    records: 342,
    time: "Just now",
  },
]

export default function BulkUploadPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-8">
      {/* Upload card */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-lg bg-primary/10">
              <CloudUpload className="size-5 text-primary" />
            </div>
            <div>
              <CardTitle>Upload Files</CardTitle>
              <CardDescription>
                Drag and drop files or click to browse. Supports CSV, JSON, XLSX,
                and plain text.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <BulkUploadForm />
        </CardContent>
      </Card>

      {/* Recent uploads */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Uploads</CardTitle>
          <CardDescription>
            Files ingested in the last 7 days
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <div className="divide-y divide-border">
            {recentUploads.map((upload) => (
              <div
                key={upload.name}
                className="flex items-center gap-4 px-6 py-4"
              >
                <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted">
                  <FileText className="size-4 text-muted-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="truncate text-sm font-medium">{upload.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {upload.size} · {upload.records.toLocaleString()} records
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  {upload.status === "completed" ? (
                    <Badge
                      variant="secondary"
                      className="gap-1 text-emerald-600"
                    >
                      <CheckCircle2 className="size-3" />
                      Completed
                    </Badge>
                  ) : (
                    <Badge variant="secondary" className="gap-1 text-amber-600">
                      <Clock className="size-3 animate-pulse" />
                      Processing
                    </Badge>
                  )}
                  <span className="hidden text-xs text-muted-foreground sm:block">
                    {upload.time}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
