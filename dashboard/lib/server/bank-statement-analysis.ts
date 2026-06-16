import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { bankStatementAnalysisSection } from "@/lib/bank-statement-analysis-config";

export type BankStatementDocumentInput = {
  id?: string;
  fileName: string;
  fileType?: string | null;
  file: Blob;
};

export type BankStatementAnalysisInput = {
  bankName: string;
  documents: BankStatementDocumentInput[];
  pdfPassword?: string;
  companyNameOverride?: string;
};

export type BankAnalysisExportFormat = "html" | "excel" | "json";

type BridgeMode = "analyze" | "export";

type BridgeArtifact = {
  fileName?: unknown;
  contentType?: unknown;
  contentBase64?: unknown;
};

type JsonRecord = Record<string, unknown>;

const BANK_LOGIC_DIR = path.join(process.cwd(), "bank-statement-analysis-logic");
const BANK_BRIDGE_SCRIPT = path.join(BANK_LOGIC_DIR, "dashboard_bridge.py");
const BANK_ANALYSIS_TIMEOUT_MS = Number(
  process.env.BANK_STATEMENT_ANALYZER_TIMEOUT_MS || 300_000
);

export class BankAnalysisError extends Error {
  code: string;
  status: number;
  detail?: unknown;

  constructor(
    code: string,
    message: string,
    status = 500,
    detail?: unknown
  ) {
    super(message);
    this.name = "BankAnalysisError";
    this.code = code;
    this.status = status;
    this.detail = detail;
  }
}

export async function runBankStatementAnalysisFromPdfs(
  input: BankStatementAnalysisInput
) {
  const supportedBanks = bankStatementAnalysisSection.tool
    .supportedBanks as readonly string[];

  if (!supportedBanks.includes(input.bankName)) {
    throw new BankAnalysisError(
      "invalid_bank_statement_bank",
      "Select a supported bank format",
      400,
      { bankName: input.bankName }
    );
  }

  if (!input.documents.length) {
    throw new BankAnalysisError(
      "missing_bank_statement_input",
      "Upload or select at least one bank statement PDF",
      400
    );
  }

  const tmpDir = mkdtempSync(path.join(os.tmpdir(), "kl-bank-analysis-"));

  try {
    const files = [];

    for (const [index, document] of input.documents.entries()) {
      const extension = getFileExtension(document.fileName);

      if (extension !== ".pdf" && document.fileType !== "application/pdf") {
        throw new BankAnalysisError(
          "invalid_bank_statement_file_type",
          "Bank Statement Analyzer only accepts PDF files",
          400,
          { fileName: document.fileName, fileType: document.fileType }
        );
      }

      const fileName = document.fileName || `bank-statement-${index + 1}.pdf`;
      const tempFileName = `${index + 1}-${sanitizeTempFileName(fileName)}`;
      const filePath = path.join(tmpDir, tempFileName);
      const buffer = Buffer.from(await document.file.arrayBuffer());

      writeFileSync(filePath, buffer);
      files.push({
        id: document.id,
        fileName,
        fileType: document.fileType || "application/pdf",
        path: filePath,
      });
    }

    return await runBridge("analyze", {
      bankName: input.bankName,
      pdfPassword: input.pdfPassword || "",
      companyNameOverride: input.companyNameOverride || "",
      files,
    });
  } finally {
    rmSync(tmpDir, { recursive: true, force: true });
  }
}

export async function exportBankAnalysisArtifact({
  format,
  report,
  fileName,
}: {
  format: BankAnalysisExportFormat;
  report: unknown;
  fileName?: string;
}) {
  if (format === "html") {
    const html = getReportHtml(report);

    if (html) {
      return {
        fileName: `${sanitizeExportBaseName(fileName || "bank-analysis")}.html`,
        contentType: "text/html; charset=utf-8",
        content: Buffer.from(html, "utf-8"),
      };
    }
  }

  if (format === "json") {
    const payload = getJsonExportPayload(report);

    return {
      fileName: `${sanitizeExportBaseName(fileName || "bank-analysis")}.json`,
      contentType: "application/json",
      content: Buffer.from(JSON.stringify(payload ?? null, null, 2), "utf-8"),
    };
  }

  const result = (await runBridge("export", {
    format,
    report,
    fileName: sanitizeExportBaseName(fileName || "bank-analysis"),
  })) as BridgeArtifact;

  const artifactFileName =
    typeof result.fileName === "string" ? result.fileName : null;
  const contentType =
    typeof result.contentType === "string" ? result.contentType : null;
  const contentBase64 =
    typeof result.contentBase64 === "string" ? result.contentBase64 : null;

  if (!artifactFileName || !contentType || !contentBase64) {
    throw new BankAnalysisError(
      "bank_statement_export_failure",
      "Bank statement export renderer returned an invalid artifact",
      502,
      result
    );
  }

  return {
    fileName: artifactFileName,
    contentType,
    content: Buffer.from(contentBase64, "base64"),
  };
}

function runBridge(mode: BridgeMode, request: unknown) {
  return new Promise<unknown>((resolve, reject) => {
    const tmpDir = mkdtempSync(path.join(os.tmpdir(), "kl-bank-bridge-"));
    const requestPath = path.join(tmpDir, "request.json");
    const outputPath = path.join(tmpDir, "output.json");
    const pythonCandidates = getPythonCandidates();
    let candidateIndex = 0;
    let settled = false;
    let lastStartError: unknown;

    writeFileSync(requestPath, JSON.stringify(request), "utf-8");

    const cleanup = () => {
      rmSync(tmpDir, { recursive: true, force: true });
    };

    const tryNext = () => {
      if (candidateIndex >= pythonCandidates.length) {
        cleanup();
        reject(
          new BankAnalysisError(
            "bank_statement_python_unavailable",
            "No Python interpreter was available to run the bank statement analyzer",
            500,
            lastStartError instanceof Error
              ? lastStartError.message
              : String(lastStartError || "")
          )
        );
        return;
      }

      const pythonBin = pythonCandidates[candidateIndex];
      candidateIndex += 1;
      const child = spawn(
        pythonBin,
        [
          BANK_BRIDGE_SCRIPT,
          mode,
          "--request",
          requestPath,
          "--output",
          outputPath,
        ],
        {
          cwd: BANK_LOGIC_DIR,
          env: {
            ...process.env,
            PYTHONUNBUFFERED: "1",
          },
          windowsHide: true,
        }
      );
      let stdout = "";
      let stderr = "";
      let startFailed = false;

      const timer = setTimeout(() => {
        child.kill("SIGKILL");
      }, BANK_ANALYSIS_TIMEOUT_MS);

      child.stdout.on("data", (chunk: Buffer) => {
        stdout += chunk.toString("utf-8");
      });
      child.stderr.on("data", (chunk: Buffer) => {
        stderr += chunk.toString("utf-8");
      });

      child.on("error", (error) => {
        clearTimeout(timer);
        startFailed = true;
        lastStartError = error;
        tryNext();
      });

      child.on("close", (code, signal) => {
        clearTimeout(timer);

        if (settled || startFailed) return;

        let output: unknown = null;
        try {
          output = JSON.parse(readFileSync(outputPath, "utf-8"));
        } catch {
          output = null;
        }

        if (signal === "SIGKILL") {
          settled = true;
          cleanup();
          reject(
            new BankAnalysisError(
              "bank_statement_analysis_timeout",
              "Bank statement analyzer timed out",
              504,
              { timeoutMs: BANK_ANALYSIS_TIMEOUT_MS }
            )
          );
          return;
        }

        if (code !== 0) {
          settled = true;
          cleanup();
          reject(
            new BankAnalysisError(
              "bank_statement_analysis_failure",
              getBridgeErrorMessage(output) || "Bank statement analyzer failed",
              502,
              {
                output,
                stdout: stdout.slice(-4000),
                stderr: stderr.slice(-4000),
              }
            )
          );
          return;
        }

        if (!isRecord(output)) {
          settled = true;
          cleanup();
          reject(
            new BankAnalysisError(
              "bank_statement_analysis_failure",
              "Bank statement analyzer returned malformed output",
              502,
              { stdout: stdout.slice(-4000), stderr: stderr.slice(-4000) }
            )
          );
          return;
        }

        settled = true;
        cleanup();
        resolve(output);
      });
    };

    tryNext();
  });
}

function getPythonCandidates() {
  return [
    process.env.BANK_STATEMENT_ANALYZER_PYTHON_BIN,
    process.env.FINANCIAL_RENDERER_PYTHON_BIN,
    process.platform === "win32" ? "py" : "python3",
    "python",
  ].filter((value): value is string => Boolean(value));
}

function getBridgeErrorMessage(output: unknown) {
  if (!isRecord(output)) return "";
  return typeof output.error === "string" ? output.error : "";
}

function getReportHtml(report: unknown) {
  if (!isRecord(report)) return "";

  if (typeof report.html === "string") return report.html;

  const nestedReport = report.report;
  if (isRecord(nestedReport) && typeof nestedReport.html === "string") {
    return nestedReport.html;
  }

  return "";
}

function getJsonExportPayload(report: unknown) {
  if (!isRecord(report)) return report;
  if (isRecord(report.full_report)) return report.full_report;
  if (isRecord(report.report_json)) return getJsonExportPayload(report.report_json);
  return report;
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function getFileExtension(fileName: string) {
  const dotIndex = fileName.lastIndexOf(".");
  return dotIndex === -1 ? "" : fileName.slice(dotIndex).toLowerCase();
}

function sanitizeTempFileName(fileName: string) {
  return (
    fileName
      .replace(/[<>:"/\\|?*\x00-\x1F]/g, "-")
      .replace(/\s+/g, "-")
      .slice(0, 120) || "bank-statement.pdf"
  );
}

function sanitizeExportBaseName(fileName: string) {
  return (
    fileName
      .replace(/[<>:"/\\|?*\x00-\x1F]/g, "-")
      .replace(/\s+/g, "-")
      .replace(/\.+$/g, "")
      .slice(0, 80) || "bank-analysis"
  );
}
