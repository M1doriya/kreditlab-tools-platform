import {
  BankAnalysisError,
  exportBankAnalysisArtifact,
  type BankAnalysisExportFormat,
} from "@/lib/server/bank-statement-analysis";

export const runtime = "nodejs";
export const maxDuration = 300;

const EXPORT_FORMATS = new Set<BankAnalysisExportFormat>([
  "html",
  "excel",
  "json",
]);

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as {
      format?: unknown;
      report?: unknown;
      fileName?: unknown;
    };
    const format =
      typeof body.format === "string" &&
      EXPORT_FORMATS.has(body.format as BankAnalysisExportFormat)
        ? (body.format as BankAnalysisExportFormat)
        : null;

    if (!format) {
      return Response.json(
        {
          error: "Export format must be html, excel, or json",
          code: "missing_input",
        },
        { status: 400 }
      );
    }

    const artifact = await exportBankAnalysisArtifact({
      format,
      report: body.report,
      fileName: typeof body.fileName === "string" ? body.fileName : undefined,
    });

    return new Response(
      new Blob([new Uint8Array(artifact.content)], {
        type: artifact.contentType,
      }),
      {
        headers: {
          "Content-Type": artifact.contentType,
          "Content-Disposition": `attachment; filename="${sanitizeHeaderFileName(
            artifact.fileName
          )}"`,
          "Cache-Control": "no-store",
        },
      }
    );
  } catch (error) {
    if (error instanceof BankAnalysisError) {
      return Response.json(
        {
          error: error.message,
          code: error.code,
          detail: error.detail,
        },
        { status: error.status }
      );
    }

    return Response.json(
      {
        error: "Failed to export bank statement analysis",
        code: "dashboard_execution_failure",
        detail: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

function sanitizeHeaderFileName(fileName: string) {
  return fileName.replace(/["\r\n]/g, "").trim() || "bank-analysis";
}
