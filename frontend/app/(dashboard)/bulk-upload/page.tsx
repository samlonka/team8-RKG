import type { Metadata } from "next"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { CloudUpload } from "lucide-react"
import { BulkUploadForm } from "./BulkUploadForm"

export const metadata: Metadata = {
  title: "Bulk Upload",
}

export default function BulkUploadPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-lg bg-primary/10">
              <CloudUpload className="size-5 text-primary" />
            </div>
            <div>
              <CardTitle>Tenant SKU Ingest</CardTitle>
              <CardDescription>
                Upload a tenant SKU export (.xlsx) to match and merge records
                against the Global Knowledge Graph. Results appear as soon as
                processing completes.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <BulkUploadForm />
        </CardContent>
      </Card>
    </div>
  )
}
