"use client";

import {
  Component,
  useEffect,
  useMemo,
  useState,
  type ErrorInfo,
  type ReactNode,
} from "react";
import {
  AlertTriangle,
  BarChart3,
  Download,
  FileText,
  Loader2,
  Trash2,
  Upload,
} from "lucide-react";
import {
  CLAUDE_MODEL_PRICING,
  DEFAULT_CLAUDE_MODEL_ID,
  estimateClaudeCost,
} from "@/lib/claude-models";
import { financialStatementAnalysisSection } from "@/lib/financial-statement-analysis-config";
import { supabase } from "@/lib/supabase";
import { ReportPreviewFrame } from "./report-preview-frame";

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

type GeneratedTextFile = {
  id: string;
  originalFileName: string;
  generatedFileName: string;
  fileType: "text/plain";
  text: string;
  textLength: number;
  ocrProvider?: string;
  ocrPagesParsed?: number;
  servedBy?: string;
};

type LocalTextFile = {
  id: string;
  fileName: string;
  text: string;
  textLength: number;
};

type ConvertFinancialPdfResponse = {
  error?: unknown;
  code?: unknown;
  detail?: unknown;
  generatedTextFiles?: GeneratedTextFile[];
};

type RunFinancialAnalysisResponse = {
  error?: unknown;
  code?: unknown;
  detail?: unknown;
  report?: unknown;
};

type FinancialAnalysisOptionsResponse = {
  defaultModel?: unknown;
};

type FinancialAnalysisEstimate = {
  model: string;
  engine: "analyze.py" | "json-passthrough";
  inputTokens: number;
  likelyOutputTokens: number;
  maxOutputTokens: number;
  likelyCostUsd: number;
  worstCostUsd: number;
};

type EstimateResponse = {
  estimate?: FinancialAnalysisEstimate;
  error?: unknown;
};

type FinancialExportFormat = "html" | "pdf" | "excel";

type ExistingTextContent = {
  text: string;
  textLength: number;
  loading: boolean;
  error?: string;
};

type FinancialStatementAnalysisSectionProps = {
  caseId: string;
  documents: CaseDocument[];
  onAnalysisSaved: (report: AnalysisReport) => void;
  onAnalysisReportsRefresh: () => Promise<void>;
};

type FinancialStatementAnalysisErrorBoundaryProps = {
  children: ReactNode;
};

type FinancialStatementAnalysisErrorBoundaryState = {
  hasError: boolean;
};

export class FinancialStatementAnalysisErrorBoundary extends Component<
  FinancialStatementAnalysisErrorBoundaryProps,
  FinancialStatementAnalysisErrorBoundaryState
> {
  state: FinancialStatementAnalysisErrorBoundaryState = {
    hasError: false,
  };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Financial Statement Analysis renderer failed", {
      stage: "renderer",
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
    });
  }

  render() {
    if (this.state.hasError) {
      return (
        <section className="rounded-2xl border border-red-200 bg-red-50 p-6 text-sm text-red-700">
          Financial Statement Analysis could not load. Check server logs for
          details.
        </section>
      );
    }

    return this.props.children;
  }
}

const isObjectRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const getErrorDetailMessage = (detail: unknown): string => {
  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    return detail
      .filter((item): item is string => typeof item === "string")
      .slice(0, 3)
      .join("; ");
  }

  if (!isObjectRecord(detail)) return "";

  const errors = detail.errors;
  if (Array.isArray(errors)) {
    return errors
      .filter((item): item is string => typeof item === "string")
      .slice(0, 3)
      .join("; ");
  }

  if (typeof detail.message === "string") return detail.message;

  const stdoutTail = detail.stdoutTail;
  if (typeof stdoutTail === "string" && stdoutTail.trim()) {
    return stdoutTail.trim().split(/\r?\n/).slice(-3).join("; ");
  }

  const stderrTail = detail.stderrTail;
  if (typeof stderrTail === "string" && stderrTail.trim()) {
    return stderrTail.trim().split(/\r?\n/).slice(-3).join("; ");
  }

  const body = detail.body;
  if (body !== detail) {
    const bodyMessage = getErrorDetailMessage(body);
    if (bodyMessage) return bodyMessage;
  }

  const nestedDetail = detail.detail;
  if (nestedDetail !== detail) {
    return getErrorDetailMessage(nestedDetail);
  }

  return "";
};

const getErrorMessage = (
  result: { error?: unknown; code?: unknown; detail?: unknown },
  fallback: string
) => {
  const base = typeof result.error === "string" ? result.error : fallback;
  const code = typeof result.code === "string" ? result.code : "";
  const detail = getErrorDetailMessage(result.detail);
  const message = detail && !base.includes(detail) ? `${base}: ${detail}` : base;

  return code ? `${message} (${code})` : message;
};

const isPdfFile = (fileName: string, fileType: string | null) =>
  fileName.toLowerCase().endsWith(".pdf") || fileType === "application/pdf";

const isTextInputFile = (fileName: string, fileType: string | null) => {
  const normalizedName = fileName.toLowerCase();

  return (
    normalizedName.endsWith(".txt") ||
    normalizedName.endsWith(".md") ||
    normalizedName.endsWith(".json") ||
    fileType === "text/plain" ||
    fileType === "text/markdown" ||
    fileType === "text/x-markdown" ||
    fileType === "application/json"
  );
};

export function FinancialStatementAnalysisSection({
  caseId,
  documents,
  onAnalysisSaved,
  onAnalysisReportsRefresh,
}: FinancialStatementAnalysisSectionProps) {
  const [uploadedPdfs, setUploadedPdfs] = useState<File[]>([]);
  const [selectedPdfDocumentIds, setSelectedPdfDocumentIds] = useState<string[]>([]);
  const [generatedTextFiles, setGeneratedTextFiles] = useState<GeneratedTextFile[]>(
    []
  );
  const [selectedGeneratedTextIds, setSelectedGeneratedTextIds] = useState<
    string[]
  >([]);
  const [localTextFiles, setLocalTextFiles] = useState<LocalTextFile[]>([]);
  const [selectedLocalTextIds, setSelectedLocalTextIds] = useState<string[]>([]);
  const [selectedExistingTextDocumentIds, setSelectedExistingTextDocumentIds] =
    useState<string[]>([]);
  const [removedExistingTextDocumentIds, setRemovedExistingTextDocumentIds] =
    useState<string[]>([]);
  const [existingTextContents, setExistingTextContents] = useState<
    Record<string, ExistingTextContent>
  >({});
  const [selectedClaudeModel, setSelectedClaudeModel] = useState<string>(
    DEFAULT_CLAUDE_MODEL_ID
  );
  const [converting, setConverting] = useState(false);
  const [runningAnalysis, setRunningAnalysis] = useState(false);
  const [conversionStatus, setConversionStatus] = useState("");
  const [analysisStatus, setAnalysisStatus] = useState("");
  const [ocrError, setOcrError] = useState("");
  const [claudeError, setClaudeError] = useState("");
  const [analysisHtml, setAnalysisHtml] = useState("");
  const [analysisReport, setAnalysisReport] = useState<unknown>(null);
  const [exportingFormat, setExportingFormat] =
    useState<FinancialExportFormat | null>(null);
  const [exportError, setExportError] = useState("");

  const existingPdfDocuments = useMemo(
    () => documents.filter((doc) => isPdfFile(doc.file_name, doc.file_type)),
    [documents]
  );
  const existingTextDocuments = useMemo(
    () =>
      documents.filter(
        (doc) =>
          isTextInputFile(doc.file_name, doc.file_type) &&
          !removedExistingTextDocumentIds.includes(doc.id)
      ),
    [documents, removedExistingTextDocumentIds]
  );
  const selectedGeneratedTextFiles = generatedTextFiles.filter((file) =>
    selectedGeneratedTextIds.includes(file.id)
  );
  const selectedLocalTextFiles = localTextFiles.filter((file) =>
    selectedLocalTextIds.includes(file.id)
  );
  const hasTextInputs =
    generatedTextFiles.length > 0 ||
    localTextFiles.length > 0 ||
    existingTextDocuments.length > 0;
  const selectedTextInputCount =
    selectedGeneratedTextFiles.length +
    selectedLocalTextFiles.length +
    selectedExistingTextDocumentIds.length;
  const selectedKnownTextLength = useMemo(() => {
    const generatedTextLength = selectedGeneratedTextFiles.reduce(
      (total, file) => total + file.text.length,
      0
    );
    const localTextLength = selectedLocalTextFiles.reduce(
      (total, file) => total + file.text.length,
      0
    );
    const existingTextLength = selectedExistingTextDocumentIds.reduce(
      (total, documentId) =>
        total + (existingTextContents[documentId]?.textLength || 0),
      0
    );

    return generatedTextLength + localTextLength + existingTextLength;
  }, [
    existingTextContents,
    selectedExistingTextDocumentIds,
    selectedGeneratedTextFiles,
    selectedLocalTextFiles,
  ]);
  const selectedExistingTextStillLoading = selectedExistingTextDocumentIds.some(
    (documentId) => existingTextContents[documentId]?.loading
  );
  const claudeCostEstimate = useMemo(
    () =>
      estimateClaudeCost({
        modelId: selectedClaudeModel,
        inputCharacters: selectedKnownTextLength,
      }),
    [selectedClaudeModel, selectedKnownTextLength]
  );

  // Exact pre-flight estimate from analyze.py --dry-run (count_tokens over the
  // real framework + docs). The chars/4 claudeCostEstimate above is only a
  // fallback shown until this resolves or if it errors.
  const [serverEstimate, setServerEstimate] = useState<{
    loading: boolean;
    data: FinancialAnalysisEstimate | null;
    error: string;
  }>({ loading: false, data: null, error: "" });

  // Stable primitive key so the debounced effect only refires on real changes,
  // not on every render (the selected* arrays are fresh objects each render).
  const estimateKey = useMemo(
    () =>
      JSON.stringify({
        model: selectedClaudeModel,
        existing: selectedExistingTextDocumentIds,
        generated: selectedGeneratedTextFiles.map((file) => [
          file.id,
          file.text.length,
        ]),
        local: selectedLocalTextFiles.map((file) => [file.id, file.text.length]),
      }),
    [
      selectedClaudeModel,
      selectedExistingTextDocumentIds,
      selectedGeneratedTextFiles,
      selectedLocalTextFiles,
    ]
  );

  useEffect(() => {
    let active = true;

    if (selectedTextInputCount === 0 || selectedExistingTextStillLoading) {
      const clear = setTimeout(() => {
        if (active) setServerEstimate({ loading: false, data: null, error: "" });
      }, 0);
      return () => {
        active = false;
        clearTimeout(clear);
      };
    }

    const timer = setTimeout(async () => {
      if (!active) return;
      setServerEstimate((prev) => ({ ...prev, loading: true, error: "" }));
      try {
        const generatedTextPayload = [
          ...selectedGeneratedTextFiles.map((file) => ({
            id: file.id,
            generatedFileName: file.generatedFileName,
            originalFileName: file.originalFileName,
            text: file.text,
          })),
          ...selectedLocalTextFiles.map((file) => ({
            id: file.id,
            generatedFileName: file.fileName,
            originalFileName: file.fileName,
            text: file.text,
          })),
        ];
        const response = await fetch(financialStatementAnalysisSection.action, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            caseId,
            documentIds: selectedExistingTextDocumentIds,
            generatedTextFiles: generatedTextPayload,
            model: selectedClaudeModel,
            mode: "estimate",
          }),
        });
        const result = (await response.json().catch(() => ({}))) as EstimateResponse;

        if (!active) return;

        if (!response.ok || !result.estimate) {
          setServerEstimate({
            loading: false,
            data: null,
            error:
              typeof result.error === "string"
                ? result.error
                : "Could not compute the exact estimate.",
          });
          return;
        }

        setServerEstimate({ loading: false, data: result.estimate, error: "" });
      } catch (error) {
        if (!active) return;
        setServerEstimate({
          loading: false,
          data: null,
          error:
            error instanceof Error
              ? error.message
              : "Could not compute the exact estimate.",
        });
      }
    }, 600);

    return () => {
      active = false;
      clearTimeout(timer);
    };
    // estimateKey encodes every input that changes the estimate; the values read
    // inside are captured fresh whenever it changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [estimateKey]);

  useEffect(() => {
    let active = true;

    const loadDefaultModel = async () => {
      try {
        const response = await fetch(financialStatementAnalysisSection.action);
        const result =
          (await response.json().catch(() => ({}))) as FinancialAnalysisOptionsResponse;
        const defaultModel =
          typeof result.defaultModel === "string" ? result.defaultModel : "";

        if (
          active &&
          CLAUDE_MODEL_PRICING.some((model) => model.id === defaultModel)
        ) {
          setSelectedClaudeModel(defaultModel);
        }
      } catch {
        // Keep the local fallback model if the options request cannot complete.
      }
    };

    void loadDefaultModel();

    return () => {
      active = false;
    };
  }, []);

  const handlePdfUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    const pdfs = files.filter((file) => isPdfFile(file.name, file.type));

    if (pdfs.length !== files.length) {
      setOcrError("Step 1 only accepts PDF files.");
    } else {
      setOcrError("");
    }

    setUploadedPdfs((current) => [...current, ...pdfs]);
    event.target.value = "";
  };

  const handleTextUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    const acceptedFiles = files.filter((file) => isTextInputFile(file.name, file.type));

    if (acceptedFiles.length !== files.length) {
      setClaudeError("Step 2 only accepts TXT, MD, or JSON files.");
    } else {
      setClaudeError("");
    }

    const loadedFiles = await Promise.all(
      acceptedFiles.map(async (file, index) => {
        const text = await file.text();

        return {
          id: `${Date.now()}-${index}-${file.name}`,
          fileName: file.name,
          text,
          textLength: text.trim().length,
        };
      })
    );

    setLocalTextFiles((current) => [...current, ...loadedFiles]);
    setSelectedLocalTextIds((current) => [
      ...current,
      ...loadedFiles.map((file) => file.id),
    ]);
    event.target.value = "";
  };

  const handleConvertWithOcr = async () => {
    if (uploadedPdfs.length === 0 && selectedPdfDocumentIds.length === 0) {
      setOcrError("Upload or select at least one PDF file.");
      return;
    }

    setConverting(true);
    setOcrError("");
    setConversionStatus("Converting PDF to TXT with Azure OCR...");
    setAnalysisHtml("");
    setAnalysisReport(null);
    setExportError("");

    try {
      const formData = new FormData();
      formData.append("caseId", caseId);
      formData.append("documentIdsJson", JSON.stringify(selectedPdfDocumentIds));

      for (const file of uploadedPdfs) {
        formData.append("files", file, file.name);
      }

      const response = await fetch(financialStatementAnalysisSection.convertAction, {
        method: "POST",
        body: formData,
      });
      const result = (await response.json().catch(() => ({
        error: "OCR conversion returned an unreadable response.",
      }))) as ConvertFinancialPdfResponse;

      if (!response.ok) {
        throw new Error(
          getErrorMessage(result, "OCR conversion failed")
        );
      }

      const newTextFiles = result.generatedTextFiles || [];

      if (newTextFiles.length === 0) {
        throw new Error("OCR conversion completed without TXT output.");
      }

      setGeneratedTextFiles((current) => [...current, ...newTextFiles]);
      setSelectedGeneratedTextIds((current) => [
        ...current,
        ...newTextFiles.map((file) => file.id),
      ]);
      setConversionStatus(
        `${newTextFiles.length} TXT file${
          newTextFiles.length === 1 ? "" : "s"
        } generated.`
      );
    } catch (conversionError) {
      setConversionStatus("");
      setOcrError(
        conversionError instanceof Error
          ? conversionError.message
          : String(conversionError)
      );
    } finally {
      setConverting(false);
    }
  };

  const removeGeneratedTextFile = (fileId: string) => {
    setGeneratedTextFiles((current) =>
      current.filter((file) => file.id !== fileId)
    );
    setSelectedGeneratedTextIds((current) =>
      current.filter((id) => id !== fileId)
    );
  };

  const removeLocalTextFile = (fileId: string) => {
    setLocalTextFiles((current) => current.filter((file) => file.id !== fileId));
    setSelectedLocalTextIds((current) => current.filter((id) => id !== fileId));
  };

  const removeExistingTextDocument = (documentId: string) => {
    setRemovedExistingTextDocumentIds((current) =>
      current.includes(documentId) ? current : [...current, documentId]
    );
    setSelectedExistingTextDocumentIds((current) =>
      current.filter((id) => id !== documentId)
    );
    setExistingTextContents((current) => {
      const remaining = { ...current };
      delete remaining[documentId];

      return remaining;
    });
  };

  const handleContinueToClaude = async () => {
    if (!hasTextInputs || selectedTextInputCount === 0) {
      setClaudeError("Select at least one TXT, MD, or JSON input first.");
      return;
    }

    setRunningAnalysis(true);
    setClaudeError("");
    setAnalysisStatus("Preparing selected inputs for analysis...");
    setAnalysisHtml("");
    setAnalysisReport(null);
    setExportError("");

    const analysisType = financialStatementAnalysisSection.tool.analysisType;

    try {
      // Baseline: the most recent existing report id, so we can recognise the
      // NEW one this run produces (vs. a report from an earlier run).
      let baselineReportId: string | null = null;
      try {
        const { data: latest } = await supabase
          .from("case_analysis_reports")
          .select("id")
          .eq("case_id", caseId)
          .eq("analysis_type", analysisType)
          .order("created_at", { ascending: false })
          .limit(1);
        baselineReportId = latest?.[0]?.id ?? null;
      } catch {
        // non-fatal — polling still works, it just keys off "newest row appears".
      }

      const generatedTextPayload = [
        ...selectedGeneratedTextFiles.map((file) => ({
          id: file.id,
          generatedFileName: file.generatedFileName,
          originalFileName: file.originalFileName,
          text: file.text,
        })),
        ...selectedLocalTextFiles.map((file) => ({
          id: file.id,
          generatedFileName: file.fileName,
          originalFileName: file.fileName,
          text: file.text,
        })),
      ];

      // Fire the run, but do NOT block on its HTTP response. A large audit runs
      // for several minutes and the connection is dropped (browser/host limits,
      // Vercel function timeout) long before it returns — the old "Failed to
      // fetch". The server keeps running and SAVES the report to the DB, so we
      // poll for that row instead. We still watch the response to surface a fast
      // validation/start error (e.g. bad input, missing API key).
      let runError: string | null = null;
      let runSettled = false;
      void fetch(financialStatementAnalysisSection.action, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          caseId,
          documentIds: selectedExistingTextDocumentIds,
          generatedTextFiles: generatedTextPayload,
          model: selectedClaudeModel,
        }),
      })
        .then(async (response) => {
          if (!response.ok) {
            const result = (await response
              .json()
              .catch(() => ({}))) as RunFinancialAnalysisResponse;
            const statusFallback = `Claude analysis failed (HTTP ${response.status}${
              response.statusText ? ` ${response.statusText}` : ""
            })`;

            runError = getErrorMessage(result, statusFallback);
          }
        })
        .catch(() => {
          // Network drop on the long request is expected — rely on polling.
        })
        .finally(() => {
          runSettled = true;
        });

      setAnalysisStatus(
        "Analyzing… this can take several minutes. You can leave this page open."
      );

      const startedAt = Date.now();
      const TIMEOUT_MS = 20 * 60 * 1000; // 20 min hard cap
      const POLL_MS = 5000;

      // Poll for the saved report.
      while (true) {
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));

        const { data: rows } = await supabase
          .from("case_analysis_reports")
          .select("id, report_html, report_json, created_at")
          .eq("case_id", caseId)
          .eq("analysis_type", analysisType)
          .order("created_at", { ascending: false })
          .limit(1);
        const row = rows?.[0];

        if (
          row &&
          row.id !== baselineReportId &&
          typeof row.report_html === "string" &&
          row.report_html
        ) {
          setAnalysisHtml(row.report_html);
          setAnalysisReport(row.report_json ?? row);
          onAnalysisSaved({
            report_html: row.report_html,
            report_json: row.report_json ?? row,
          });
          setAnalysisStatus("Financial analysis saved to this case.");
          await onAnalysisReportsRefresh();
          return;
        }

        // The run reported a fast error and no report was produced.
        if (runSettled && runError) {
          throw new Error(runError);
        }

        if (Date.now() - startedAt > TIMEOUT_MS) {
          throw new Error(
            "Analysis is taking longer than expected. It may still finish — check the saved reports for this case shortly."
          );
        }
      }
    } catch (analysisError) {
      setAnalysisStatus("");
      setClaudeError(
        analysisError instanceof Error ? analysisError.message : String(analysisError)
      );
    } finally {
      setRunningAnalysis(false);
    }
  };

  const toggleExistingTextDocument = async (doc: CaseDocument) => {
    const isSelected = selectedExistingTextDocumentIds.includes(doc.id);

    if (isSelected) {
      setSelectedExistingTextDocumentIds((current) =>
        current.filter((id) => id !== doc.id)
      );
      return;
    }

    setSelectedExistingTextDocumentIds((current) => [...current, doc.id]);

    if (existingTextContents[doc.id]) {
      return;
    }

    setExistingTextContents((current) => ({
      ...current,
      [doc.id]: { text: "", textLength: 0, loading: true },
    }));

    const { data, error } = await supabase.storage
      .from("case-documents")
      .download(doc.file_path);

    if (error || !data) {
      setExistingTextContents((current) => ({
        ...current,
        [doc.id]: {
          text: "",
          textLength: 0,
          loading: false,
          error: error?.message || "Could not estimate saved text file length",
        },
      }));
      return;
    }

    const text = await data.text();

    setExistingTextContents((current) => ({
      ...current,
      [doc.id]: {
        text,
        textLength: text.trim().length,
        loading: false,
      },
    }));
  };

  const toggleValue = (
    value: string,
    setValues: React.Dispatch<React.SetStateAction<string[]>>
  ) => {
    setValues((current) =>
      current.includes(value)
        ? current.filter((item) => item !== value)
        : [...current, value]
    );
  };

  const downloadGeneratedTextFile = (file: GeneratedTextFile | LocalTextFile) => {
    const fileName =
      "generatedFileName" in file ? file.generatedFileName : file.fileName;
    const blob = new Blob([file.text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(url);
  };

  const downloadAnalysisExport = async (format: FinancialExportFormat) => {
    if (!analysisReport) {
      setExportError("Run analysis first before downloading files.");
      return;
    }

    setExportingFormat(format);
    setExportError("");

    try {
      const response = await fetch("/api/export-financial-analysis", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          format,
          report: analysisReport,
        }),
      });

      if (!response.ok) {
        const result = (await response.json().catch(() => ({}))) as {
          error?: unknown;
          code?: unknown;
        };
        throw new Error(getErrorMessage(result, "Financial analysis export failed"));
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
      id={financialStatementAnalysisSection.id}
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-4">
          <div className="rounded-xl bg-cyan-50 p-3">
            <BarChart3 className="h-6 w-6 text-cyan-500" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-900">
              {financialStatementAnalysisSection.label}
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              {financialStatementAnalysisSection.tool.displayName}
            </p>
          </div>
        </div>

        <span className="rounded-full bg-cyan-50 px-3 py-1 text-xs font-semibold text-cyan-700">
          Staged PDF to TXT to analysis
        </span>
      </div>

      <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">
              Step 1: Convert PDF to TXT
            </h3>
            <p className="mt-1 text-sm text-slate-600">
              Azure OCR runs here. Claude waits for Step 2.
            </p>
          </div>

          <label className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-900 shadow-sm hover:bg-slate-50">
            <Upload className="h-4 w-4" />
            Upload PDF
            <input
              type="file"
              accept="application/pdf,.pdf"
              multiple
              onChange={handlePdfUpload}
              disabled={converting || runningAnalysis}
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
                  <div
                    key={`${file.name}-${index}`}
                    className="flex items-center justify-between rounded-lg border border-slate-200 px-3 py-2 text-sm"
                  >
                    <span className="min-w-0 truncate text-slate-700">
                      {file.name}
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        setUploadedPdfs((current) =>
                          current.filter((_, itemIndex) => itemIndex !== index)
                        )
                      }
                      className="text-xs text-slate-500 hover:text-red-600"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-sm font-semibold text-slate-900">
              Saved case PDFs
            </p>
            {existingPdfDocuments.length === 0 ? (
              <p className="mt-3 text-sm text-slate-500">
                No saved PDF financial statements found.
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
                      checked={selectedPdfDocumentIds.includes(doc.id)}
                      onChange={() =>
                        toggleValue(doc.id, setSelectedPdfDocumentIds)
                      }
                      className="h-4 w-4 rounded border-slate-300 text-cyan-500"
                    />
                    <span className="min-w-0 truncate text-slate-700">
                      {doc.file_name}
                    </span>
                  </label>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <button
            type="button"
            onClick={handleConvertWithOcr}
            disabled={
              converting ||
              runningAnalysis ||
              (uploadedPdfs.length === 0 && selectedPdfDocumentIds.length === 0)
            }
            className="inline-flex items-center justify-center rounded-xl bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-900 hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {converting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {converting ? "Converting..." : "Convert with Azure OCR"}
          </button>

          {conversionStatus && (
            <span className="text-sm font-medium text-cyan-700">
              {conversionStatus}
            </span>
          )}
        </div>

        {ocrError && (
          <ErrorMessage message={ocrError} />
        )}

        {generatedTextFiles.length > 0 && (
          <div className="mt-5 rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-sm font-semibold text-slate-900">
              Generated TXT files
            </p>
            <div className="mt-3 space-y-2">
              {generatedTextFiles.map((file) => (
                <TextFileRow
                  key={file.id}
                  fileName={file.generatedFileName}
                  detail={`${file.textLength.toLocaleString()} characters from ${
                    file.originalFileName
                  }`}
                  checked={selectedGeneratedTextIds.includes(file.id)}
                  onCheckedChange={() =>
                    toggleValue(file.id, setSelectedGeneratedTextIds)
                  }
                  onDownload={() => downloadGeneratedTextFile(file)}
                  onRemove={() => removeGeneratedTextFile(file.id)}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">
              Step 2: Analyze TXT / JSON
            </h3>
            <p className="mt-1 text-sm text-slate-600">
              Select generated TXT files or existing TXT, MD, and JSON inputs.
            </p>
          </div>

          <label className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-900 shadow-sm hover:bg-slate-50">
            <FileText className="h-4 w-4" />
            Add TXT / MD / JSON
            <input
              type="file"
              accept=".txt,.md,.json,text/plain,text/markdown,application/json"
              multiple
              onChange={handleTextUpload}
              disabled={runningAnalysis}
              className="hidden"
            />
          </label>
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-sm font-semibold text-slate-900">
              Generated or uploaded text
            </p>
            {generatedTextFiles.length === 0 && localTextFiles.length === 0 ? (
              <p className="mt-3 text-sm text-slate-500">
                Convert PDFs or add TXT, MD, or JSON files first.
              </p>
            ) : (
              <div className="mt-3 space-y-2">
                {generatedTextFiles.map((file) => (
                  <TextFileRow
                    key={file.id}
                    fileName={file.generatedFileName}
                    detail={`${file.textLength.toLocaleString()} characters`}
                    checked={selectedGeneratedTextIds.includes(file.id)}
                    onCheckedChange={() =>
                      toggleValue(file.id, setSelectedGeneratedTextIds)
                    }
                    onDownload={() => downloadGeneratedTextFile(file)}
                    onRemove={() => removeGeneratedTextFile(file.id)}
                  />
                ))}
                {localTextFiles.map((file) => (
                  <TextFileRow
                    key={file.id}
                    fileName={file.fileName}
                    detail={`${file.textLength.toLocaleString()} characters`}
                    checked={selectedLocalTextIds.includes(file.id)}
                    onCheckedChange={() =>
                      toggleValue(file.id, setSelectedLocalTextIds)
                    }
                    onDownload={() => downloadGeneratedTextFile(file)}
                    onRemove={() => removeLocalTextFile(file.id)}
                  />
                ))}
              </div>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-sm font-semibold text-slate-900">
              Saved text inputs
            </p>
            {existingTextDocuments.length === 0 ? (
              <p className="mt-3 text-sm text-slate-500">
                No saved TXT, MD, or JSON financial statements found.
              </p>
            ) : (
              <div className="mt-3 space-y-2">
                {existingTextDocuments.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm hover:bg-slate-50"
                  >
                    <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-3">
                      <input
                        type="checkbox"
                        checked={selectedExistingTextDocumentIds.includes(doc.id)}
                        onChange={() => void toggleExistingTextDocument(doc)}
                        className="h-4 w-4 rounded border-slate-300 text-cyan-500"
                      />
                      <span className="min-w-0 flex-1 truncate text-slate-700">
                        {doc.file_name}
                      </span>
                      {existingTextContents[doc.id]?.loading && (
                        <Loader2 className="h-4 w-4 animate-spin text-slate-400" />
                      )}
                      {existingTextContents[doc.id]?.error && (
                        <span className="shrink-0 text-xs text-red-600">
                          Estimate unavailable
                        </span>
                      )}
                    </label>
                    <button
                      type="button"
                      onClick={() => removeExistingTextDocument(doc.id)}
                      disabled={runningAnalysis}
                      className="shrink-0 rounded-lg border border-slate-200 p-2 text-slate-500 hover:border-red-200 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-60"
                      title="Remove from Stage 2"
                      aria-label={`Remove ${doc.file_name || "saved text input"}`}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(280px,360px)]">
          <div>
            <label
              htmlFor="financial-claude-model"
              className="text-sm font-semibold text-slate-900"
            >
              Claude model
            </label>
            <select
              id="financial-claude-model"
              value={selectedClaudeModel}
              onChange={(event) => setSelectedClaudeModel(event.target.value)}
              disabled={runningAnalysis}
              className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-100"
            >
              {CLAUDE_MODEL_PRICING.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-sm font-semibold text-slate-900">
              Estimated Claude cost
            </p>
            {serverEstimate.data ? (
              <>
                <p className="mt-2 text-2xl font-semibold text-slate-900">
                  {formatUsd(serverEstimate.data.worstCostUsd)}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {serverEstimate.data.inputTokens.toLocaleString()} input tokens
                  {" · "}likely {formatUsd(serverEstimate.data.likelyCostUsd)}
                  {" · "}max output{" "}
                  {serverEstimate.data.maxOutputTokens.toLocaleString()} tokens
                </p>
                <p className="mt-1 text-[11px] text-slate-400">
                  {serverEstimate.data.engine === "json-passthrough"
                    ? "Renderer-ready JSON — no Claude call."
                    : "Exact count via analyze.py (count_tokens); worst case shown."}
                </p>
              </>
            ) : (
              <>
                <p className="mt-2 text-2xl font-semibold text-slate-900">
                  {formatUsd(claudeCostEstimate.totalCostUsd)}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {claudeCostEstimate.inputTokens.toLocaleString()} input tokens
                  and {claudeCostEstimate.outputTokens.toLocaleString()} max
                  output tokens
                </p>
                <p className="mt-1 text-[11px] text-slate-400">
                  Rough estimate (chars ÷ 4)
                </p>
              </>
            )}
            {serverEstimate.loading && (
              <p className="mt-2 text-xs text-cyan-700">
                Calculating exact tokens…
              </p>
            )}
            {serverEstimate.error && (
              <p className="mt-2 text-xs text-amber-600">
                {serverEstimate.error}
              </p>
            )}
            {selectedExistingTextStillLoading && (
              <p className="mt-2 text-xs text-cyan-700">
                Updating saved text file estimate...
              </p>
            )}
          </div>
        </div>

        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <button
            type="button"
            onClick={handleContinueToClaude}
            disabled={
              runningAnalysis ||
              !hasTextInputs ||
              selectedTextInputCount === 0
            }
            className="inline-flex items-center justify-center rounded-xl bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-900 hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {runningAnalysis && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {runningAnalysis
              ? "Analyzing..."
              : "Run Analysis"}
          </button>

          {analysisStatus && (
            <span className="text-sm font-medium text-cyan-700">
              {analysisStatus}
            </span>
          )}
        </div>

        {claudeError && <ErrorMessage message={claudeError} />}
      </div>

      {analysisHtml && (
        <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <h3 className="text-lg font-semibold text-slate-900">
              Financial Analysis Result
            </h3>

            <div className="flex flex-wrap gap-2">
              <ExportButton
                label="HTML"
                format="html"
                exportingFormat={exportingFormat}
                onClick={downloadAnalysisExport}
              />
              <ExportButton
                label="PDF"
                format="pdf"
                exportingFormat={exportingFormat}
                onClick={downloadAnalysisExport}
              />
              <ExportButton
                label="Excel"
                format="excel"
                exportingFormat={exportingFormat}
                onClick={downloadAnalysisExport}
              />
            </div>
          </div>

          {exportError && <ErrorMessage message={exportError} />}

          <ReportPreviewFrame
            title="Financial statement analysis result"
            html={analysisHtml}
            className="mt-4"
          />
        </div>
      )}
    </section>
  );
}

function ExportButton({
  label,
  format,
  exportingFormat,
  onClick,
}: {
  label: string;
  format: FinancialExportFormat;
  exportingFormat: FinancialExportFormat | null;
  onClick: (format: FinancialExportFormat) => void;
}) {
  const exporting = exportingFormat === format;
  const disabled = exportingFormat !== null;

  return (
    <button
      type="button"
      onClick={() => onClick(format)}
      disabled={disabled}
      className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
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

function TextFileRow({
  fileName,
  detail,
  checked,
  onCheckedChange,
  onDownload,
  onRemove,
}: {
  fileName: string;
  detail: string;
  checked: boolean;
  onCheckedChange: () => void;
  onDownload: () => void;
  onRemove?: () => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm hover:bg-slate-50">
      <div className="flex min-w-0 items-center gap-3">
        <input
          type="checkbox"
          checked={checked}
          onChange={onCheckedChange}
          className="h-4 w-4 rounded border-slate-300 text-cyan-500"
        />
        <div className="min-w-0">
          <p className="truncate font-medium text-slate-900">{fileName}</p>
          <p className="text-xs text-slate-500">{detail}</p>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-2">
        <button
          type="button"
          onClick={(event) => {
            event.preventDefault();
            onDownload();
          }}
          className="rounded-lg border border-slate-200 p-2 text-slate-500 hover:text-slate-900"
          aria-label={`Download ${fileName}`}
        >
          <Download className="h-4 w-4" />
        </button>

        {onRemove && (
          <button
            type="button"
            onClick={(event) => {
              event.preventDefault();
              onRemove();
            }}
            className="rounded-lg border border-slate-200 p-2 text-slate-500 hover:border-red-200 hover:text-red-600"
            aria-label={`Remove ${fileName}`}
          >
            <Trash2 className="h-4 w-4" />
          </button>
        )}
      </div>
    </label>
  );
}

function formatUsd(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: value >= 1 ? 2 : 4,
    maximumFractionDigits: value >= 1 ? 2 : 4,
  }).format(value);
}

function getDownloadFileName(contentDisposition: string) {
  const match = contentDisposition.match(/filename="([^"]+)"/i);

  return match?.[1] || "";
}

function getDefaultExportFileName(format: FinancialExportFormat) {
  switch (format) {
    case "pdf":
      return "financial-analysis.pdf";
    case "excel":
      return "financial-analysis.xlsx";
    default:
      return "financial-analysis.html";
  }
}

function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="mt-5 flex gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}
