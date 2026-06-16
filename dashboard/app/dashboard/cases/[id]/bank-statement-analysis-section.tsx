"use client";

import {
  Component,
  useMemo,
  useState,
  type ChangeEvent,
  type Dispatch,
  type ErrorInfo,
  type ReactNode,
  type SetStateAction,
} from "react";
import {
  AlertTriangle,
  Download,
  Landmark,
  Loader2,
  Trash2,
  Upload,
} from "lucide-react";
import {
  bankStatementAnalysisSection,
  type BankStatementBankName,
} from "@/lib/bank-statement-analysis-config";

type CaseDocument = {
  id: string;
  case_id: string;
  file_name: string;
  file_path: string;
  file_type: string | null;
  document_type: string | null;
  uploaded_at: string | null;
};

type AnalysisReport = {
  report_html: string;
  report_json: unknown;
};

type BankStatementAnalysisSectionProps = {
  caseId: string;
  documents: CaseDocument[];
  onAnalysisSaved: (report: AnalysisReport) => void;
  onAnalysisReportsRefresh: () => Promise<void>;
};

type BankStatementAnalysisResponse = {
  error?: unknown;
  code?: unknown;
  detail?: unknown;
  report?: unknown;
  savedReport?: AnalysisReport;
};

type BankExportFormat = "html" | "excel" | "json";

type BankStatementAnalysisErrorBoundaryProps = {
  children: ReactNode;
};

type BankStatementAnalysisErrorBoundaryState = {
  hasError: boolean;
};

export class BankStatementAnalysisErrorBoundary extends Component<
  BankStatementAnalysisErrorBoundaryProps,
  BankStatementAnalysisErrorBoundaryState
> {
  state: BankStatementAnalysisErrorBoundaryState = {
    hasError: false,
  };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Bank Statement Analysis renderer failed", {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
    });
  }

  render() {
    if (this.state.hasError) {
      return (
        <section className="rounded-2xl border border-red-200 bg-red-50 p-6 text-sm text-red-700">
          Bank Statement Analysis could not load. Check server logs for details.
        </section>
      );
    }

    return this.props.children;
  }
}

const isObjectRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

export function BankStatementAnalysisSection({
  caseId,
  documents,
  onAnalysisSaved,
  onAnalysisReportsRefresh,
}: BankStatementAnalysisSectionProps) {
  const [bankName, setBankName] = useState<BankStatementBankName>(
    bankStatementAnalysisSection.tool.supportedBanks[0]
  );
  const [uploadedPdfs, setUploadedPdfs] = useState<File[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [pdfPassword, setPdfPassword] = useState("");
  const [companyNameOverride, setCompanyNameOverride] = useState("");
  const [runningAnalysis, setRunningAnalysis] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState("");
  const [analysisError, setAnalysisError] = useState("");
  const [analysisHtml, setAnalysisHtml] = useState("");
  const [analysisReport, setAnalysisReport] = useState<unknown>(null);
  const [exportingFormat, setExportingFormat] = useState<BankExportFormat | null>(
    null
  );
  const [exportError, setExportError] = useState("");

  const existingPdfDocuments = useMemo(
    () => documents.filter((doc) => isPdfFile(doc.file_name, doc.file_type)),
    [documents]
  );

  const selectedInputCount = uploadedPdfs.length + selectedDocumentIds.length;

  const handlePdfUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []).filter((file) =>
      isPdfFile(file.name, file.type)
    );

    setUploadedPdfs((current) => [...current, ...files]);
    event.target.value = "";
  };

  const handleRunAnalysis = async () => {
    if (selectedInputCount === 0) {
      setAnalysisError("Upload or select at least one bank statement PDF.");
      return;
    }

    setRunningAnalysis(true);
    setAnalysisStatus("Preparing bank statement PDFs...");
    setAnalysisError("");
    setExportError("");

    try {
      const formData = new FormData();
      formData.append("caseId", caseId);
      formData.append("bankName", bankName);
      formData.append("documentIdsJson", JSON.stringify(selectedDocumentIds));

      if (pdfPassword.trim()) {
        formData.append("pdfPassword", pdfPassword.trim());
      }

      if (companyNameOverride.trim()) {
        formData.append("companyNameOverride", companyNameOverride.trim());
      }

      for (const file of uploadedPdfs) {
        formData.append("files", file, file.name);
      }

      setAnalysisStatus("Running bank parser and renderer...");
      const response = await fetch(bankStatementAnalysisSection.action, {
        method: "POST",
        body: formData,
      });
      const result = (await response.json().catch(() => ({}))) as
        BankStatementAnalysisResponse;

      if (!response.ok) {
        throw new Error(getErrorMessage(result, "Bank statement analysis failed"));
      }

      const reportHtml = getReportHtml(result.report) || result.savedReport?.report_html;

      if (!reportHtml) {
        throw new Error("Bank statement analysis did not return report HTML.");
      }

      setAnalysisHtml(reportHtml);
      setAnalysisReport(result.report || result.savedReport?.report_json || null);

      if (result.savedReport) {
        onAnalysisSaved(result.savedReport);
      } else {
        onAnalysisSaved({
          report_html: reportHtml,
          report_json: result.report || null,
        });
      }

      await onAnalysisReportsRefresh();
      setAnalysisStatus("Bank statement analysis saved to this case.");
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : String(error));
      setAnalysisStatus("");
    } finally {
      setRunningAnalysis(false);
    }
  };

  const downloadAnalysisExport = async (format: BankExportFormat) => {
    if (!analysisReport) {
      setExportError("Run analysis first before downloading files.");
      return;
    }

    setExportingFormat(format);
    setExportError("");

    try {
      const response = await fetch(bankStatementAnalysisSection.exportAction, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          format,
          report: analysisReport,
          fileName: "bank-statement-analysis",
        }),
      });

      if (!response.ok) {
        const result = (await response.json().catch(() => ({}))) as {
          error?: unknown;
          code?: unknown;
        };
        throw new Error(getErrorMessage(result, "Bank statement export failed"));
      }

      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") || "";
      const fileName =
        getDownloadFileName(disposition) || getDefaultExportFileName(format);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");

      link.href = url;
      link.download = fileName;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : String(error));
    } finally {
      setExportingFormat(null);
    }
  };

  return (
    <section
      id={bankStatementAnalysisSection.id}
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-4">
          <div className="rounded-xl bg-emerald-50 p-3">
            <Landmark className="h-6 w-6 text-emerald-600" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-900">
              {bankStatementAnalysisSection.label}
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              {bankStatementAnalysisSection.tool.displayName}
            </p>
          </div>
        </div>

        <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">
          Direct PDF parser and report
        </span>
      </div>

      <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-5">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(260px,340px)]">
          <div>
            <label
              htmlFor="bank-statement-bank"
              className="text-sm font-semibold text-slate-900"
            >
              Bank format
            </label>
            <select
              id="bank-statement-bank"
              value={bankName}
              onChange={(event) =>
                setBankName(
                  event.target.value as BankStatementBankName
                )
              }
              disabled={runningAnalysis}
              className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
            >
              {bankStatementAnalysisSection.tool.supportedBanks.map((bank) => (
                <option key={bank} value={bank}>
                  {bank}
                </option>
              ))}
            </select>
          </div>

          <label className="inline-flex cursor-pointer items-center justify-center gap-2 self-end rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-900 shadow-sm hover:bg-slate-50">
            <Upload className="h-4 w-4" />
            Upload PDF
            <input
              type="file"
              accept={bankStatementAnalysisSection.tool.accept}
              multiple
              onChange={handlePdfUpload}
              disabled={runningAnalysis}
              className="hidden"
            />
          </label>
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-sm font-semibold text-slate-900">Uploaded PDFs</p>
            {uploadedPdfs.length === 0 ? (
              <p className="mt-3 text-sm text-slate-500">No uploaded PDFs yet.</p>
            ) : (
              <div className="mt-3 space-y-2">
                {uploadedPdfs.map((file, index) => (
                  <FileRow
                    key={`${file.name}-${index}`}
                    name={file.name}
                    detail={`${formatFileSize(file.size)}`}
                    checked
                    disabled={runningAnalysis}
                    onRemove={() =>
                      setUploadedPdfs((current) =>
                        current.filter((_, itemIndex) => itemIndex !== index)
                      )
                    }
                  />
                ))}
              </div>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-sm font-semibold text-slate-900">Saved case PDFs</p>
            {existingPdfDocuments.length === 0 ? (
              <p className="mt-3 text-sm text-slate-500">
                No saved PDF bank statements found.
              </p>
            ) : (
              <div className="mt-3 space-y-2">
                {existingPdfDocuments.map((doc) => (
                  <label
                    key={doc.id}
                    className="flex cursor-pointer items-center gap-3 rounded-lg border border-slate-200 px-3 py-2 text-sm hover:bg-slate-50"
                  >
                    <input
                      type="checkbox"
                      checked={selectedDocumentIds.includes(doc.id)}
                      onChange={() => toggleValue(doc.id, setSelectedDocumentIds)}
                      disabled={runningAnalysis}
                      className="h-4 w-4 rounded border-slate-300 text-emerald-600"
                    />
                    <span className="min-w-0 flex-1 truncate text-slate-700">
                      {doc.file_name}
                    </span>
                  </label>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <div>
            <label
              htmlFor="bank-pdf-password"
              className="text-sm font-semibold text-slate-900"
            >
              PDF password
            </label>
            <input
              id="bank-pdf-password"
              type="password"
              value={pdfPassword}
              onChange={(event) => setPdfPassword(event.target.value)}
              disabled={runningAnalysis}
              placeholder="Only needed for encrypted PDFs"
              className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
            />
          </div>

          <div>
            <label
              htmlFor="bank-company-override"
              className="text-sm font-semibold text-slate-900"
            >
              Company name override
            </label>
            <input
              id="bank-company-override"
              value={companyNameOverride}
              onChange={(event) => setCompanyNameOverride(event.target.value)}
              disabled={runningAnalysis}
              placeholder="Optional"
              className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
            />
          </div>
        </div>

        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <button
            type="button"
            onClick={handleRunAnalysis}
            disabled={runningAnalysis || selectedInputCount === 0}
            className="inline-flex items-center justify-center rounded-xl bg-emerald-500 px-5 py-3 text-sm font-semibold text-white hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {runningAnalysis && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {runningAnalysis ? "Analyzing..." : "Run Bank Analysis"}
          </button>

          {analysisStatus && (
            <span className="text-sm font-medium text-emerald-700">
              {analysisStatus}
            </span>
          )}
        </div>

        {analysisError && <ErrorMessage message={analysisError} />}
      </div>

      {analysisHtml && (
        <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <h3 className="text-lg font-semibold text-slate-900">
              Bank Statement Analysis Result
            </h3>

            <div className="flex flex-wrap gap-2">
              <ExportButton
                label="HTML"
                format="html"
                exportingFormat={exportingFormat}
                onClick={downloadAnalysisExport}
              />
              <ExportButton
                label="Excel"
                format="excel"
                exportingFormat={exportingFormat}
                onClick={downloadAnalysisExport}
              />
              <ExportButton
                label="JSON"
                format="json"
                exportingFormat={exportingFormat}
                onClick={downloadAnalysisExport}
              />
            </div>
          </div>

          {exportError && <ErrorMessage message={exportError} />}

          <div
            className="mt-4 max-h-[900px] overflow-auto rounded-xl border border-slate-200"
            dangerouslySetInnerHTML={{
              __html: analysisHtml,
            }}
          />
        </div>
      )}
    </section>
  );
}

function FileRow({
  name,
  detail,
  checked,
  disabled,
  onRemove,
}: {
  name: string;
  detail: string;
  checked: boolean;
  disabled: boolean;
  onRemove: () => void;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-slate-200 px-3 py-2 text-sm">
      <input
        type="checkbox"
        checked={checked}
        readOnly
        className="h-4 w-4 rounded border-slate-300 text-emerald-600"
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-slate-700">{name}</p>
        <p className="text-xs text-slate-400">{detail}</p>
      </div>
      <button
        type="button"
        onClick={onRemove}
        disabled={disabled}
        className="shrink-0 rounded-lg border border-slate-200 p-2 text-slate-500 hover:border-red-200 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-60"
        title="Remove PDF"
        aria-label={`Remove ${name}`}
      >
        <Trash2 className="h-4 w-4" />
      </button>
    </div>
  );
}

function ExportButton({
  label,
  format,
  exportingFormat,
  onClick,
}: {
  label: string;
  format: BankExportFormat;
  exportingFormat: BankExportFormat | null;
  onClick: (format: BankExportFormat) => void;
}) {
  const exporting = exportingFormat === format;

  return (
    <button
      type="button"
      onClick={() => onClick(format)}
      disabled={exportingFormat !== null}
      className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
    >
      {exporting ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        <Download className="h-4 w-4" />
      )}
      {exporting ? "Preparing..." : `Save ${label}`}
    </button>
  );
}

function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="mt-4 flex gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <p>{message}</p>
    </div>
  );
}

function toggleValue(
  value: string,
  setter: Dispatch<SetStateAction<string[]>>
) {
  setter((current) =>
    current.includes(value)
      ? current.filter((item) => item !== value)
      : [...current, value]
  );
}

function getReportHtml(report: unknown) {
  if (!isObjectRecord(report)) return "";
  if (typeof report.html === "string") return report.html;
  if (isObjectRecord(report.report) && typeof report.report.html === "string") {
    return report.report.html;
  }
  return "";
}

function getErrorMessage(
  result: { error?: unknown; code?: unknown; detail?: unknown },
  fallback: string
) {
  const message = typeof result.error === "string" ? result.error : fallback;
  const code = typeof result.code === "string" ? result.code : "";
  const detail = getErrorDetailMessage(result.detail);
  const base = code ? `${message} (${code})` : message;

  return detail ? `${base}: ${detail}` : base;
}

function getErrorDetailMessage(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (!isObjectRecord(detail)) return "";
  if (typeof detail.message === "string") return detail.message;
  if (typeof detail.error === "string") return detail.error;
  return "";
}

function getDownloadFileName(contentDisposition: string) {
  const match = contentDisposition.match(/filename="([^"]+)"/i);
  return match?.[1] || "";
}

function getDefaultExportFileName(format: BankExportFormat) {
  switch (format) {
    case "excel":
      return "bank-statement-analysis.xlsx";
    case "json":
      return "bank-statement-analysis.json";
    default:
      return "bank-statement-analysis.html";
  }
}

function isPdfFile(fileName: string, fileType: string | null) {
  return fileName.toLowerCase().endsWith(".pdf") || fileType === "application/pdf";
}

function formatFileSize(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}
