import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import {
  estimateClaudeCost,
  getClaudeModelById,
  resolveClaudeEffort,
  resolveClaudeModelId,
} from "@/lib/claude-models";

type JsonRecord = Record<string, unknown>;

type FinancialStatementSourceKind =
  | "ocr_markdown"
  | "markdown"
  | "text"
  | "json";

type ExtractedFinancialStatement = {
  fileName: string;
  fileType: string | null;
  sourceKind: FinancialStatementSourceKind;
  markdown?: string;
  analysisJson?: JsonRecord;
  ocr?: {
    fileId?: string;
    parseId?: string;
    pagesParsed?: number;
    provider?: string;
    servedBy?: string;
  };
};

type ClaudeAnalysisResult = {
  json: JsonRecord;
  model: string;
  usage?: unknown;
};

type ClaudeMessagesResult = {
  text: string;
  usage?: unknown;
  stopReason?: string;
  stopSequence?: string;
};

type FinancialStageLogInput = {
  stage: "renderer" | "ocr_conversion" | "claude_analysis";
  files?: Array<{
    fileName: string;
    fileType: string | null;
    fileSize?: number;
  }>;
  generatedTextLength?: number;
  claudeInputTextLength?: number;
  claudeModel?: string;
  error?: unknown;
  extra?: JsonRecord;
};

type LogicRendererValidation = {
  isValid: boolean;
  errors: string[];
  warnings: string[];
};

type LogicRendererResult = {
  success?: unknown;
  validation?: unknown;
  html?: unknown;
  contentBase64?: unknown;
  contentType?: unknown;
  fileExtension?: unknown;
  error?: unknown;
  traceback?: unknown;
};

type LogicRendererMode = "validate" | "render" | "pdf" | "excel";

type PythonRendererCandidate = {
  command: string;
  argsPrefix: string[];
  source: string;
};

type RendererServiceCandidate = {
  baseUrl: string;
  source: string;
};

export type FinancialStatementDocumentInput = {
  id?: string;
  fileName: string;
  fileType: string | null;
  file: Blob;
};

export type FinancialStatementTextInput = {
  id?: string;
  fileName: string;
  fileType: string | null;
  text: string;
  sourceFileName?: string;
  sourceKind?: "generated_txt" | "text" | "markdown";
};

export type FinancialStatementJsonInput = {
  id?: string;
  fileName: string;
  fileType: string | null;
  json: JsonRecord;
};

export type FinancialStatementConversionResult = {
  success: true;
  tool: "financial_statement";
  generatedTextFiles: Array<{
    id: string;
    originalFileName: string;
    generatedFileName: string;
    fileType: "text/plain";
    text: string;
    textLength: number;
    ocrProvider?: string;
    ocrPagesParsed?: number;
    servedBy?: string;
  }>;
};

export type FinancialStatementAnalysisResult = {
  success: true;
  tool: "financial_statement";
  html: string;
  json: JsonRecord;
  warnings: string[];
  reports: Array<{
    file_name: string;
    source_file_type: string | null;
    html: string;
    json: JsonRecord;
    warnings: string[];
  }>;
  extraction: Array<{
    file_name: string;
    source_kind: FinancialStatementSourceKind;
    ocr_provider?: string;
    ocr_pages_parsed?: number;
    served_by?: string;
  }>;
  claude?: {
    model: string;
    usage?: unknown;
  };
};

export type FinancialAnalysisExportFormat = "html" | "pdf" | "excel";

export type FinancialAnalysisExportArtifact = {
  content: ArrayBuffer;
  contentType: string;
  fileName: string;
};

export type FinancialAnalysisErrorCode =
  | "missing_input"
  | "missing_claude_api_key"
  | "invalid_file_type"
  | "pdf_upload_failure"
  | "ocr_extraction_failure"
  | "ocr_empty_output"
  | "text_file_empty"
  | "invalid_claude_model"
  | "claude_auth_failure"
  | "claude_model_not_found"
  | "claude_rate_limit"
  | "claude_context_too_large"
  | "claude_output_truncated"
  | "invalid_claude_request"
  | "claude_analysis_failure"
  | "missing_financial_statement_input"
  | "malformed_intermediate_output"
  | "dashboard_execution_failure";

export class FinancialAnalysisError extends Error {
  code: FinancialAnalysisErrorCode;
  status: number;
  detail?: unknown;

  constructor(
    code: FinancialAnalysisErrorCode,
    message: string,
    status: number,
    detail?: unknown
  ) {
    super(message);
    this.name = "FinancialAnalysisError";
    this.code = code;
    this.status = status;
    this.detail = detail;
  }
}

const ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";
const DEFAULT_ANTHROPIC_MAX_TOKENS = 64000;
const DEFAULT_AZURE_OCR_TIMEOUT_MS = 240000;
const DEFAULT_RENDERER_TIMEOUT_MS = 60000;
const DEFAULT_FINANCIAL_RENDERER_API_URL =
  "https://financial-statement-analysis.kreditlab.my";
const FINANCIAL_LOGIC_DIR = path.join(
  process.cwd(),
  "financial-statement-analysis-logic"
);
const FINANCIAL_LOGIC_BRIDGE = path.join(FINANCIAL_LOGIC_DIR, "render_bridge.py");
const FINANCIAL_AZURE_OCR_SCRIPT = path.join(FINANCIAL_LOGIC_DIR, "azure_ocr.py");
const CLAUDE_SCHEMA_INSTRUCTIONS_FILE = path.join(
  FINANCIAL_LOGIC_DIR,
  "claude_schema_instructions.md"
);
const FINANCIAL_ANALYZE_SCRIPT = path.join(FINANCIAL_LOGIC_DIR, "analyze.py");
// Generous ceiling so large audits (e.g. MTC's ~200KB doc) finish like they do
// in the standalone CLI, which has no timeout. Only a genuinely hung process
// should ever hit this. Override with FINANCIAL_ANALYZE_TIMEOUT_MS.
const DEFAULT_ANALYZE_TIMEOUT_MS = 1200000; // 20 min
// The pre-flight estimate only counts tokens (free count_tokens endpoint, no
// Claude call), so it returns in ~1s. Override with FINANCIAL_ESTIMATE_TIMEOUT_MS.
const DEFAULT_ESTIMATE_TIMEOUT_MS = 60000; // 1 min

const CLAUDE_SYSTEM_PROMPT = loadClaudeSchemaInstructions();

function loadClaudeSchemaInstructions() {
  try {
    const instructions = readFileSync(
      CLAUDE_SCHEMA_INSTRUCTIONS_FILE,
      "utf8"
    ).trim();

    if (!instructions) {
      throw new Error("schema instruction file is empty");
    }

    return instructions;
  } catch (error) {
    throw new Error(
      `Could not load Claude schema instructions from ${CLAUDE_SCHEMA_INSTRUCTIONS_FILE}: ${
        error instanceof Error ? error.message : String(error)
      }`
    );
  }
}

export async function convertFinancialPdfsToText(
  documents: FinancialStatementDocumentInput[]
): Promise<FinancialStatementConversionResult> {
  if (!documents.length) {
    throw new FinancialAnalysisError("missing_input", "No PDF files selected", 400);
  }

  logFinancialStage({
    stage: "ocr_conversion",
    files: documents.map(getDocumentLogInfo),
    extra: {
      hasAzureDocumentIntelligenceConfig: hasAzureDocumentIntelligenceConfig(),
      hasServiceApiKey: Boolean(process.env.SERVICE_API_KEY),
    },
  });

  const generatedTextFiles: FinancialStatementConversionResult["generatedTextFiles"] =
    [];

  for (const document of documents) {
    if (!document.file || !document.fileName) {
      throw new FinancialAnalysisError(
        "missing_input",
        "Missing PDF input",
        400,
        { fileName: document.fileName }
      );
    }

    if (!isPdfDocument(document)) {
      throw new FinancialAnalysisError(
        "invalid_file_type",
        "Step 1 only accepts PDF files",
        400,
        { fileName: document.fileName, fileType: document.fileType }
      );
    }

    const ocrResult = await extractPdfWithAzureOcr(document);
    const text = ocrResult.markdown.trim();

    if (!text) {
      throw new FinancialAnalysisError(
        "ocr_empty_output",
        "Azure OCR returned empty text",
        502,
        { fileName: document.fileName, provider: ocrResult.provider }
      );
    }

    generatedTextFiles.push({
      id: `${Date.now()}-${generatedTextFiles.length}-${slugifyFileName(document.fileName)}`,
      originalFileName: document.fileName,
      generatedFileName: getGeneratedTextFileName(document.fileName),
      fileType: "text/plain",
      text,
      textLength: text.length,
      ocrProvider: ocrResult.provider,
      ocrPagesParsed: ocrResult.pagesParsed,
      servedBy: ocrResult.servedBy,
    });
  }

  logFinancialStage({
    stage: "ocr_conversion",
    files: documents.map(getDocumentLogInfo),
    generatedTextLength: generatedTextFiles.reduce(
      (total, file) => total + file.textLength,
      0
    ),
    extra: {
      generatedFileCount: generatedTextFiles.length,
    },
  });

  return {
    success: true,
    tool: "financial_statement",
    generatedTextFiles,
  };
}

export async function runFinancialStatementAnalysisFromText(
  input: {
    textDocuments?: FinancialStatementTextInput[];
    jsonDocuments?: FinancialStatementJsonInput[];
    model?: string;
  }
): Promise<FinancialStatementAnalysisResult> {
  const textDocuments = input.textDocuments || [];
  const jsonDocuments = input.jsonDocuments || [];

  if (textDocuments.length === 0 && jsonDocuments.length === 0) {
    throw new FinancialAnalysisError(
      "missing_input",
      "Select at least one TXT, MD, or JSON input",
      400
    );
  }

  const markdownInputs = textDocuments.map((document) => {
    const text = document.text.trim();

    if (!text) {
      throw new FinancialAnalysisError(
        "text_file_empty",
        "Financial statement text is empty",
        400,
        { fileName: document.fileName }
      );
    }

    return {
      fileName: document.fileName,
      fileType: document.fileType,
      sourceKind:
        document.sourceKind === "markdown"
          ? ("markdown" as FinancialStatementSourceKind)
          : ("text" as FinancialStatementSourceKind),
      markdown: text,
    };
  });

  if (markdownInputs.length > 0 && jsonDocuments.length > 0) {
    throw new FinancialAnalysisError(
      "missing_input",
      "Select TXT/MD source inputs for Claude or one renderer-compatible JSON report, not both",
      400
    );
  }

  if (jsonDocuments.length > 1) {
    throw new FinancialAnalysisError(
      "missing_input",
      "Select only one renderer-compatible JSON report at a time",
      400
    );
  }

  const primaryDocument = markdownInputs[0] || jsonDocuments[0];
  let analysisJson: JsonRecord;
  let claude: ClaudeAnalysisResult | undefined;

  if (markdownInputs.length > 0) {
    const anthropicApiKey = getAnthropicApiKey();
    const model = resolveClaudeModelId(input.model, process.env.ANTHROPIC_MODEL);

    if (!model) {
      throw new FinancialAnalysisError(
        "invalid_claude_model",
        "Selected Claude model is not allowed",
        400,
        { model: input.model }
      );
    }

    const claudeInputTextLength = markdownInputs.reduce(
      (total, document) => total + document.markdown.length,
      0
    );
    const maxOutputTokens = getClaudeMaxOutputTokens(model);
    const costEstimate = estimateClaudeCost({
      modelId: model,
      inputCharacters: claudeInputTextLength,
      outputTokens: maxOutputTokens,
    });

    logFinancialStage({
      stage: "claude_analysis",
      files: markdownInputs.map((document) => ({
        fileName: document.fileName,
        fileType: document.fileType,
        fileSize: document.markdown.length,
      })),
      claudeInputTextLength,
      claudeModel: model,
      extra: {
        hasAnthropicApiKey: Boolean(process.env.ANTHROPIC_API_KEY),
        hasClaudeApiKey: Boolean(process.env.CLAUDE_API_KEY),
        estimatedInputTokens: costEstimate.inputTokens,
        estimatedOutputTokens: costEstimate.outputTokens,
        estimatedCostUsd: costEstimate.totalCostUsd,
      },
    });

    if (!anthropicApiKey) {
      throw new FinancialAnalysisError(
        "missing_claude_api_key",
        "ANTHROPIC_API_KEY or CLAUDE_API_KEY is missing",
        500
      );
    }

    claude = await analyzeMarkdownWithAnalyzePy(
      markdownInputs,
      model,
      anthropicApiKey
    );
    analysisJson = claude.json;
  } else {
    analysisJson = jsonDocuments[0].json;
  }

  const renderer = await renderFinancialAnalysisWithLogic(analysisJson);
  const validation = renderer.validation;
  const html = renderer.html;
  const primaryFileName = primaryDocument?.fileName || "financial-analysis.txt";
  const primaryFileType = primaryDocument?.fileType || "text/plain";

  return {
    success: true,
    tool: "financial_statement",
    html,
    json: analysisJson,
    warnings: validation.warnings,
    reports: [
      {
        file_name: primaryFileName,
        source_file_type: primaryFileType,
        html,
        json: analysisJson,
        warnings: validation.warnings,
      },
    ],
    extraction: [
      ...markdownInputs.map((item) => ({
        file_name: item.fileName,
        source_kind: item.sourceKind,
      })),
      ...jsonDocuments.map((item) => ({
        file_name: item.fileName,
        source_kind: "json" as FinancialStatementSourceKind,
      })),
    ],
    claude: claude
      ? {
          model: claude.model,
          usage: claude.usage,
        }
      : undefined,
  };
}

export async function runFinancialStatementAnalysis(
  documents: FinancialStatementDocumentInput[]
): Promise<FinancialStatementAnalysisResult> {
  if (!documents.length) {
    throw new FinancialAnalysisError(
      "missing_financial_statement_input",
      "Missing financial statement input",
      400
    );
  }

  const primaryDocument = documents[0];

  if (!primaryDocument) {
    throw new FinancialAnalysisError(
      "missing_financial_statement_input",
      "Missing financial statement input",
      400
    );
  }

  const extracted = await extractFinancialStatements(documents);
  validateExtractedStatements(extracted);

  const markdownInputs = extracted.filter(
    (item): item is ExtractedFinancialStatement & { markdown: string } =>
      typeof item.markdown === "string" && item.markdown.trim().length > 0
  );

  const jsonInputs = extracted.filter(
    (item): item is ExtractedFinancialStatement & { analysisJson: JsonRecord } =>
      isRecord(item.analysisJson)
  );

  let analysisJson: JsonRecord;
  let claude: ClaudeAnalysisResult | undefined;

  if (markdownInputs.length > 0) {
    const anthropicApiKey = getAnthropicApiKey();
    const model = resolveClaudeModelId(null, process.env.ANTHROPIC_MODEL);

    if (!model) {
      throw new FinancialAnalysisError(
        "invalid_claude_model",
        "Default Claude model is not allowed",
        400,
        { model: process.env.ANTHROPIC_MODEL }
      );
    }

    if (!anthropicApiKey) {
      throw new FinancialAnalysisError(
        "missing_claude_api_key",
        "ANTHROPIC_API_KEY or CLAUDE_API_KEY is missing",
        500
      );
    }

    claude = await analyzeMarkdownWithAnalyzePy(
      markdownInputs,
      model,
      anthropicApiKey
    );
    analysisJson = claude.json;
  } else if (jsonInputs.length > 0) {
    analysisJson = jsonInputs[0].analysisJson;
  } else {
    throw new FinancialAnalysisError(
      "malformed_intermediate_output",
      "Extractor did not return markdown or JSON analysis data",
      502,
      extracted.map((item) => ({
        fileName: item.fileName,
        sourceKind: item.sourceKind,
      }))
    );
  }

  const renderer = await renderFinancialAnalysisWithLogic(analysisJson);
  const validation = renderer.validation;
  const html = renderer.html;

  return {
    success: true,
    tool: "financial_statement",
    html,
    json: analysisJson,
    warnings: validation.warnings,
    reports: [
      {
        file_name: primaryDocument.fileName,
        source_file_type: primaryDocument.fileType,
        html,
        json: analysisJson,
        warnings: validation.warnings,
      },
    ],
    extraction: extracted.map((item) => ({
      file_name: item.fileName,
      source_kind: item.sourceKind,
      ocr_provider: item.ocr?.provider,
      ocr_pages_parsed: item.ocr?.pagesParsed,
      served_by: item.ocr?.servedBy,
    })),
    claude: claude
      ? {
          model: claude.model,
          usage: claude.usage,
        }
      : undefined,
  };
}

async function extractFinancialStatements(
  documents: FinancialStatementDocumentInput[]
): Promise<ExtractedFinancialStatement[]> {
  const extracted: ExtractedFinancialStatement[] = [];

  for (const document of documents) {
    if (!document.file || !document.fileName) {
      throw new FinancialAnalysisError(
        "missing_financial_statement_input",
        "Missing financial statement input",
        400,
        { fileName: document.fileName }
      );
    }

    const extension = getFileExtension(document.fileName);

    if (extension === ".json" || isJsonMimeType(document.fileType)) {
      const text = await readBlobAsText(document.file, document.fileName);
      const analysisJson = parseUploadedAnalysisJson(text, document.fileName);
      extracted.push({
        fileName: document.fileName,
        fileType: document.fileType,
        sourceKind: "json",
        analysisJson,
      });
      continue;
    }

    if (extension === ".txt" || isTextMimeType(document.fileType)) {
      const markdown = (await readBlobAsText(document.file, document.fileName)).trim();

      if (!markdown) {
        throw new FinancialAnalysisError(
          "missing_financial_statement_input",
          "Financial statement text is empty",
          400,
          { fileName: document.fileName }
        );
      }

      extracted.push({
        fileName: document.fileName,
        fileType: document.fileType,
        sourceKind: "text",
        markdown,
      });
      continue;
    }

    if (extension === ".md" || isMarkdownMimeType(document.fileType)) {
      const markdown = (await readBlobAsText(document.file, document.fileName)).trim();

      if (!markdown) {
        throw new FinancialAnalysisError(
          "missing_financial_statement_input",
          "Financial statement markdown is empty",
          400,
          { fileName: document.fileName }
        );
      }

      extracted.push({
        fileName: document.fileName,
        fileType: document.fileType,
        sourceKind: "markdown",
        markdown,
      });
      continue;
    }

    if (extension === ".pdf" || isPdfMimeType(document.fileType)) {
      const ocrResult = await extractPdfWithAzureOcr(document);

      extracted.push({
        fileName: document.fileName,
        fileType: document.fileType,
        sourceKind: "ocr_markdown",
        markdown: ocrResult.markdown,
        ocr: {
          fileId: ocrResult.fileId,
          parseId: ocrResult.parseId,
          pagesParsed: ocrResult.pagesParsed,
          provider: ocrResult.provider,
          servedBy: ocrResult.servedBy,
        },
      });
      continue;
    }

    throw new FinancialAnalysisError(
      "missing_financial_statement_input",
      "Unsupported financial statement input. Upload PDF, TXT, MD, or JSON.",
      400,
      { fileName: document.fileName, fileType: document.fileType }
    );
  }

  return extracted;
}

function normalizeFinancialAnalysisReportForExport(report: unknown): {
  json: JsonRecord;
} {
  const json = findReportJson(report);

  return {
    json,
  };
}

function findReportJson(value: unknown): JsonRecord {
  if (!isRecord(value)) {
    throw new FinancialAnalysisError(
      "malformed_intermediate_output",
      "Financial analysis report is missing JSON data",
      400
    );
  }

  if (isLikelyFinancialAnalysisJson(value)) return value;

  for (const key of ["json", "report_json", "report"]) {
    const child = value[key];
    if (!isRecord(child)) continue;

    try {
      return findReportJson(child);
    } catch {
      // Keep looking through other common report shapes.
    }
  }

  const reports = value.reports;

  if (Array.isArray(reports)) {
    for (const report of reports) {
      try {
        return findReportJson(report);
      } catch {
        // Keep looking through other reports.
      }
    }
  }

  throw new FinancialAnalysisError(
    "malformed_intermediate_output",
    "Financial analysis report is missing JSON data",
    400
  );
}

function isLikelyFinancialAnalysisJson(value: JsonRecord) {
  return (
    "company_info" in value ||
    "statement_of_comprehensive_income" in value ||
    "statement_of_financial_position" in value ||
    "financial_ratios" in value ||
    "company" in value ||
    "income_statement" in value ||
    "balance_sheet" in value
  );
}

function getExportCompanyName(data: JsonRecord) {
  const company = getCompanyInfo(data);

  return (
    getStringValue(company.legal_name) ||
    getStringValue(company.name) ||
    "financial-analysis"
  );
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);

  return copy.buffer;
}

async function extractPdfWithAzureOcr(
  document: FinancialStatementDocumentInput
) {
  const timeoutMs = DEFAULT_AZURE_OCR_TIMEOUT_MS;

  if (!hasAzureDocumentIntelligenceConfig()) {
    throw new FinancialAnalysisError(
      "ocr_extraction_failure",
      "Azure Document Intelligence is not configured",
      500,
      {
        fileName: document.fileName,
        hasAzureDocumentIntelligenceEndpoint: Boolean(
          process.env.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
        ),
        hasAzureDocumentIntelligenceKey: Boolean(
          process.env.AZURE_DOCUMENT_INTELLIGENCE_KEY
        ),
      }
    );
  }

  return extractPdfWithAzureDocumentIntelligence(document, timeoutMs, {
    reason: "direct_azure_document_intelligence",
  });
}

async function extractPdfWithAzureDocumentIntelligence(
  document: FinancialStatementDocumentInput,
  timeoutMs: number,
  trigger: JsonRecord
) {
  const tmpDir = mkdtempSync(path.join(os.tmpdir(), "kl-azure-ocr-"));
  const inputPath = path.join(
    tmpDir,
    sanitizeTempFileName(document.fileName || "input.pdf")
  );
  const outPath = path.join(tmpDir, "ocr-result.json");
  const failures: JsonRecord[] = [];

  try {
    const bytes = Buffer.from(await document.file.arrayBuffer());
    writeFileSync(inputPath, bytes);

    for (const candidate of getPythonRendererCandidates()) {
      try {
        const responseBody = await runAzureOcrPyCandidate(
          candidate,
          inputPath,
          outPath,
          timeoutMs
        );
        const markdown = extractMarkdownFromOcrResult(responseBody);
        const pagesParsed =
          getNumberFromPath(responseBody, ["parsed_pages_count"]) ??
          getNumberFromPath(responseBody, ["usage", "pages_parsed"]) ??
          undefined;
        const servedBy = getStringFromRecord(responseBody, "served_by") || "azure";
        const provider =
          getStringFromRecord(responseBody, "provider") || servedBy || "azure";

        if (!markdown.trim()) {
          throw new FinancialAnalysisError(
            "ocr_empty_output",
            "Azure OCR did not return markdown content",
            502,
            { fileName: document.fileName, trigger }
          );
        }

        return {
          fileId: undefined,
          parseId: undefined,
          markdown,
          pagesParsed,
          provider,
          servedBy,
        };
      } catch (error) {
        failures.push({
          candidate: candidate.command,
          source: candidate.source,
          detail:
            error instanceof FinancialAnalysisError
              ? error.detail || error.message
              : error instanceof Error
                ? error.message
                : String(error),
        });
      }
    }

    throw new FinancialAnalysisError(
      "ocr_extraction_failure",
      "Azure OCR could not be started",
      502,
      { fileName: document.fileName, trigger, failures }
    );
  } finally {
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {
      // best-effort cleanup; ignore
    }
  }
}

function sanitizeTempFileName(fileName: string) {
  const sanitized = fileName.replace(/[^\w.-]+/g, "_").replace(/^_+|_+$/g, "");
  return sanitized || "document.pdf";
}

function runAzureOcrPyCandidate(
  candidate: PythonRendererCandidate,
  inputPath: string,
  outPath: string,
  timeoutMs: number
): Promise<JsonRecord> {
  return new Promise((resolve, reject) => {
    const args = [
      ...candidate.argsPrefix,
      FINANCIAL_AZURE_OCR_SCRIPT,
      inputPath,
      "--out",
      outPath,
    ];

    logFinancialStage({
      stage: "ocr_conversion",
      extra: {
        engine: "azure_ocr.py",
        pythonCandidate: candidate.command,
        pythonCandidateSource: candidate.source,
      },
    });

    const child = spawn(candidate.command, args, {
      cwd: FINANCIAL_LOGIC_DIR,
      env: { ...process.env },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill();
      reject(
        new FinancialAnalysisError(
          "ocr_extraction_failure",
          "Azure OCR timed out",
          504,
          { timeoutMs }
        )
      );
    }, timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      reject(
        new FinancialAnalysisError(
          "ocr_extraction_failure",
          "Azure OCR could not be started",
          502,
          {
            startFailure: true,
            candidate: candidate.command,
            source: candidate.source,
            message: error instanceof Error ? error.message : String(error),
            code: isRecord(error) ? error.code : undefined,
          }
        )
      );
    });

    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);

      if (code !== 0) {
        reject(
          new FinancialAnalysisError(
            "ocr_extraction_failure",
            "Azure OCR failed",
            502,
            {
              code,
              stdoutTail: stdout.trim().split(/\r?\n/).slice(-8).join("\n"),
              stderrTail: stderr.trim().split(/\r?\n/).slice(-8).join("\n"),
            }
          )
        );
        return;
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(readFileSync(outPath, "utf8"));
      } catch (error) {
        reject(
          new FinancialAnalysisError(
            "ocr_extraction_failure",
            "Azure OCR returned invalid JSON",
            502,
            {
              parseError: error instanceof Error ? error.message : String(error),
              stdout,
              stderr,
            }
          )
        );
        return;
      }

      if (!isRecord(parsed)) {
        reject(
          new FinancialAnalysisError(
            "ocr_extraction_failure",
            "Azure OCR returned a non-object result",
            502,
            { code }
          )
        );
        return;
      }

      resolve(parsed as JsonRecord);
    });
  });
}

function extractMarkdownFromOcrResult(result: unknown) {
  if (!isRecord(result)) return "";

  const chunks = Array.isArray(result.chunks) ? result.chunks : [];
  const chunkMarkdown = chunks
    .map((chunk) =>
      isRecord(chunk) && typeof chunk.content === "string"
        ? chunk.content.trim()
        : ""
    )
    .filter(Boolean);

  if (chunkMarkdown.length > 0) {
    return chunkMarkdown.join("\n\n");
  }

  const pages = Array.isArray(result.pages) ? result.pages : [];

  return pages
    .map((page) => {
      if (!isRecord(page)) return "";

      const pageNumber =
        typeof page.page_number === "number"
          ? `## Page ${page.page_number}\n`
          : "";
      const fragments = Array.isArray(page.page_fragments)
        ? page.page_fragments
        : [];
      const content = fragments
        .map((fragment) =>
          isRecord(fragment) && typeof fragment.content === "string"
            ? fragment.content.trim()
            : ""
        )
        .filter(Boolean)
        .join("\n\n");

      return `${pageNumber}${content}`.trim();
    })
    .filter(Boolean)
    .join("\n\n");
}

// Runs the standalone KreditLab analyze.py engine (full v7.9 framework system
// prompt + self-correction loop) instead of re-implementing the Claude call in
// TypeScript. This is the "truest copy" of the standalone product: same prompt,
// same validators, same model defaults — so the output JSON matches the schema
// the Python renderer expects, eliminating the prose-vs-object drift the
// shrunken in-TS prompt produced.
async function analyzeMarkdownWithAnalyzePy(
  documents: Array<ExtractedFinancialStatement & { markdown: string }>,
  model: string,
  apiKey: string
): Promise<ClaudeAnalysisResult> {
  const candidates = getPythonRendererCandidates();

  if (!candidates.length) {
    throw new FinancialAnalysisError(
      "claude_analysis_failure",
      "No Python interpreter was available to run analyze.py",
      502,
      { reason: "No Python candidates were configured" }
    );
  }

  const tmpDir = mkdtempSync(path.join(os.tmpdir(), "kl-analyze-"));
  const inputPaths: string[] = [];

  documents.forEach((document, index) => {
    const baseName = (document.fileName || `document_${index}`).replace(
      /[^a-zA-Z0-9._-]/g,
      "_"
    );
    const fileName = /\.(txt|md)$/i.test(baseName)
      ? baseName
      : `${baseName}.txt`;
    const inputPath = path.join(
      tmpDir,
      `${String(index).padStart(2, "0")}_${fileName}`
    );
    writeFileSync(inputPath, document.markdown, "utf8");
    inputPaths.push(inputPath);
  });

  const outPath = path.join(tmpDir, "analysis.json");
  const timeoutMs = getPositiveNumberEnv(
    "FINANCIAL_ANALYZE_TIMEOUT_MS",
    DEFAULT_ANALYZE_TIMEOUT_MS
  );
  const failures: JsonRecord[] = [];

  try {
    for (const candidate of candidates) {
      try {
        const json = await runAnalyzePyCandidate(
          candidate,
          inputPaths,
          outPath,
          model,
          apiKey,
          timeoutMs
        );
        return { json, model };
      } catch (error) {
        if (
          error instanceof FinancialAnalysisError &&
          isRendererStartFailure(error.detail)
        ) {
          failures.push({
            candidate: candidate.command,
            source: candidate.source,
            detail: error.detail,
          });
          continue;
        }

        throw error;
      }
    }

    throw new FinancialAnalysisError(
      "claude_analysis_failure",
      "Could not start analyze.py with any Python interpreter",
      502,
      { logicDir: FINANCIAL_LOGIC_DIR, script: FINANCIAL_ANALYZE_SCRIPT, failures }
    );
  } finally {
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {
      // best-effort cleanup; ignore
    }
  }
}

function runAnalyzePyCandidate(
  candidate: PythonRendererCandidate,
  inputPaths: string[],
  outPath: string,
  model: string,
  apiKey: string,
  timeoutMs: number
): Promise<JsonRecord> {
  return new Promise((resolve, reject) => {
    const args = [
      ...candidate.argsPrefix,
      FINANCIAL_ANALYZE_SCRIPT,
      ...inputPaths,
      "--out",
      outPath,
      "--model",
      model,
      "--strict",
      "--no-thinking",
      "--confirm",
      "--max-retries",
      "2",
    ];

    logFinancialStage({
      stage: "claude_analysis",
      claudeModel: model,
      extra: {
        engine: "analyze.py",
        pythonCandidate: candidate.command,
        pythonCandidateSource: candidate.source,
        inputCount: inputPaths.length,
      },
    });

    const child = spawn(candidate.command, args, {
      cwd: FINANCIAL_LOGIC_DIR,
      env: { ...process.env, ANTHROPIC_API_KEY: apiKey },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill();
      reject(
        new FinancialAnalysisError(
          "claude_analysis_failure",
          "Financial statement analysis engine (analyze.py) timed out",
          504,
          { timeoutMs }
        )
      );
    }, timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      reject(
        new FinancialAnalysisError(
          "claude_analysis_failure",
          "Financial statement analysis engine (analyze.py) could not be started",
          502,
          {
            startFailure: true,
            candidate: candidate.command,
            source: candidate.source,
            message: error instanceof Error ? error.message : String(error),
            code: isRecord(error) ? error.code : undefined,
          }
        )
      );
    });

    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);

      // analyze.py writes the JSON to --out on exit 0 (clean) and exit 6
      // (validation warnings remain but JSON is still produced). Other exit
      // codes (3 cost-gate, 4 parse-fail, 5 refusal) leave no output file.
      let parsed: unknown;

      try {
        parsed = JSON.parse(readFileSync(outPath, "utf8"));
      } catch (error) {
        reject(
          createAnalyzePyFailureError(
            code,
            stdout,
            stderr,
            error instanceof Error ? error.message : String(error)
          )
        );
        return;
      }

      if (!isRecord(parsed)) {
        reject(
          new FinancialAnalysisError(
            "claude_analysis_failure",
            "Financial statement analysis engine (analyze.py) returned a non-object result",
            502,
            { code }
          )
        );
        return;
      }

      resolve(parsed as JsonRecord);
    });
  });
}

function createAnalyzePyFailureError(
  exitCode: number | null,
  stdout: string,
  stderr: string,
  readError: string
) {
  const message =
    getAnalyzePyFailureMessage(stdout, stderr) ||
    "Financial statement analysis engine (analyze.py) did not produce valid JSON";
  const code = getAnalyzePyFailureCode(message);

  return new FinancialAnalysisError(
    code,
    message,
    getAnalyzePyFailureHttpStatus(code),
    {
      exitCode,
      message,
      stdoutTail: stdout.slice(-2000),
      stderrTail: stderr.slice(-2000),
      readError,
    }
  );
}

function getAnalyzePyFailureMessage(stdout: string, stderr: string) {
  const lines = `${stderr}\n${stdout}`
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !/^\.+$/.test(line));
  const priorityLine = lines.find((line) => {
    const normalized = line.toLowerCase();

    return (
      normalized.startsWith("[fatal]") ||
      normalized.startsWith("[!]") ||
      normalized.includes("error code:") ||
      normalized.includes("authentication_error") ||
      normalized.includes("rate_limit_error") ||
      normalized.includes("invalid_request_error") ||
      normalized.includes("not_found_error")
    );
  });
  const message = priorityLine || lines.slice(-3).join(" | ");

  return message.replace(/^\[fatal\]\s*/i, "").slice(0, 1000);
}

function getAnalyzePyFailureCode(
  message: string
): FinancialAnalysisErrorCode {
  const normalized = message.toLowerCase();

  if (normalized.includes("unknown model")) return "invalid_claude_model";
  if (
    normalized.includes("authentication_error") ||
    normalized.includes("x-api-key") ||
    normalized.includes("api key") ||
    normalized.includes("error code: 401") ||
    normalized.includes("error code: 403")
  ) {
    return "claude_auth_failure";
  }
  if (
    normalized.includes("not_found_error") ||
    normalized.includes("model was not found") ||
    normalized.includes("error code: 404")
  ) {
    return "claude_model_not_found";
  }
  if (
    normalized.includes("rate_limit_error") ||
    normalized.includes("rate limit") ||
    normalized.includes("error code: 429")
  ) {
    return "claude_rate_limit";
  }
  if (
    normalized.includes("context") ||
    normalized.includes("too long") ||
    normalized.includes("maximum") ||
    normalized.includes("error code: 413")
  ) {
    return "claude_context_too_large";
  }
  if (
    normalized.includes("max_tokens") ||
    normalized.includes("truncated")
  ) {
    return "claude_output_truncated";
  }
  if (
    normalized.includes("invalid_request_error") ||
    normalized.includes("error code: 400")
  ) {
    return "invalid_claude_request";
  }

  return "claude_analysis_failure";
}

function getAnalyzePyFailureHttpStatus(code: FinancialAnalysisErrorCode) {
  switch (code) {
    case "invalid_claude_model":
    case "invalid_claude_request":
      return 400;
    case "claude_rate_limit":
      return 429;
    case "claude_context_too_large":
      return 413;
    case "claude_auth_failure":
    case "claude_model_not_found":
    case "claude_output_truncated":
    default:
      return 502;
  }
}

export type FinancialAnalysisCostEstimate = {
  model: string;
  // "analyze.py" = a real Claude call will happen; "json-passthrough" = a
  // renderer-ready JSON was supplied, so no Claude call and no cost.
  engine: "analyze.py" | "json-passthrough";
  inputTokens: number;
  likelyOutputTokens: number;
  maxOutputTokens: number;
  likelyCostUsd: number;
  worstCostUsd: number;
};

// Pre-flight cost estimate that mirrors EXACTLY what a run will spend, by asking
// the same analyze.py engine the run uses for its token count via --dry-run
// (Anthropic's free count_tokens endpoint over the full framework + docs payload).
// This replaces the crude chars/4 heuristic, which ignored the framework system
// prompt and under-counted dense financial text — see estimateClaudeCost.
export async function estimateFinancialStatementAnalysisCostFromText(input: {
  textDocuments?: FinancialStatementTextInput[];
  jsonDocuments?: FinancialStatementJsonInput[];
  model?: string;
}): Promise<FinancialAnalysisCostEstimate> {
  const textDocuments = input.textDocuments || [];

  const model = resolveClaudeModelId(input.model, process.env.ANTHROPIC_MODEL);

  if (!model) {
    throw new FinancialAnalysisError(
      "invalid_claude_model",
      "Selected Claude model is not allowed",
      400,
      { model: input.model }
    );
  }

  const markdownInputs = textDocuments
    .map((document) => ({
      fileName: document.fileName,
      markdown: document.text.trim(),
    }))
    .filter((document) => document.markdown.length > 0);

  // A renderer-ready JSON report skips Claude entirely (see the run flow), so it
  // costs nothing. Report zero rather than spawning analyze.py for nothing.
  if (markdownInputs.length === 0) {
    return {
      model,
      engine: "json-passthrough",
      inputTokens: 0,
      likelyOutputTokens: 0,
      maxOutputTokens: 0,
      likelyCostUsd: 0,
      worstCostUsd: 0,
    };
  }

  const anthropicApiKey = getAnthropicApiKey();

  if (!anthropicApiKey) {
    throw new FinancialAnalysisError(
      "missing_claude_api_key",
      "ANTHROPIC_API_KEY or CLAUDE_API_KEY is missing",
      500
    );
  }

  return estimateAnalysisCostWithAnalyzePy(markdownInputs, model, anthropicApiKey);
}

async function estimateAnalysisCostWithAnalyzePy(
  documents: Array<{ fileName?: string; markdown: string }>,
  model: string,
  apiKey: string
): Promise<FinancialAnalysisCostEstimate> {
  const candidates = getPythonRendererCandidates();

  if (!candidates.length) {
    throw new FinancialAnalysisError(
      "claude_analysis_failure",
      "No Python interpreter was available to run analyze.py",
      502,
      { reason: "No Python candidates were configured" }
    );
  }

  const tmpDir = mkdtempSync(path.join(os.tmpdir(), "kl-estimate-"));
  const inputPaths: string[] = [];

  documents.forEach((document, index) => {
    const baseName = (document.fileName || `document_${index}`).replace(
      /[^a-zA-Z0-9._-]/g,
      "_"
    );
    const fileName = /\.(txt|md)$/i.test(baseName) ? baseName : `${baseName}.txt`;
    const inputPath = path.join(
      tmpDir,
      `${String(index).padStart(2, "0")}_${fileName}`
    );
    writeFileSync(inputPath, document.markdown, "utf8");
    inputPaths.push(inputPath);
  });

  const timeoutMs = getPositiveNumberEnv(
    "FINANCIAL_ESTIMATE_TIMEOUT_MS",
    DEFAULT_ESTIMATE_TIMEOUT_MS
  );
  const failures: JsonRecord[] = [];

  try {
    for (const candidate of candidates) {
      try {
        return await runAnalyzePyDryRunCandidate(
          candidate,
          inputPaths,
          model,
          apiKey,
          timeoutMs
        );
      } catch (error) {
        if (
          error instanceof FinancialAnalysisError &&
          isRendererStartFailure(error.detail)
        ) {
          failures.push({
            candidate: candidate.command,
            source: candidate.source,
            detail: error.detail,
          });
          continue;
        }

        throw error;
      }
    }

    throw new FinancialAnalysisError(
      "claude_analysis_failure",
      "Could not start analyze.py with any Python interpreter",
      502,
      { logicDir: FINANCIAL_LOGIC_DIR, script: FINANCIAL_ANALYZE_SCRIPT, failures }
    );
  } finally {
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {
      // best-effort cleanup; ignore
    }
  }
}

// Parses the two summary lines analyze.py prints during --dry-run, e.g.:
//   [tokens] input: 92,767  likely output: ~12,000  max output: 64,000
//   [cost]   likely: $0.8798   worst (max_tokens hit): $2.1798
function parseDryRunEstimate(
  stdout: string,
  model: string
): FinancialAnalysisCostEstimate | null {
  const toNumber = (value: string) => Number(value.replace(/,/g, ""));

  const tokens = stdout.match(
    /input:\s*([\d,]+)\s+likely output:\s*~?([\d,]+)\s+max output:\s*([\d,]+)/i
  );
  const cost = stdout.match(/likely:\s*\$([\d.]+)\s+worst[^$]*\$([\d.]+)/i);

  if (!tokens || !cost) {
    return null;
  }

  return {
    model,
    engine: "analyze.py",
    inputTokens: toNumber(tokens[1]),
    likelyOutputTokens: toNumber(tokens[2]),
    maxOutputTokens: toNumber(tokens[3]),
    likelyCostUsd: Number(cost[1]),
    worstCostUsd: Number(cost[2]),
  };
}

function runAnalyzePyDryRunCandidate(
  candidate: PythonRendererCandidate,
  inputPaths: string[],
  model: string,
  apiKey: string,
  timeoutMs: number
): Promise<FinancialAnalysisCostEstimate> {
  return new Promise((resolve, reject) => {
    // Mirror the input-token-affecting flags the real run uses (runAnalyzePyCandidate):
    // same model, same --strict system prompt, default --max-tokens (64000). Only
    // then does the count match what the run will actually send.
    const args = [
      ...candidate.argsPrefix,
      FINANCIAL_ANALYZE_SCRIPT,
      ...inputPaths,
      "--dry-run",
      "--model",
      model,
      "--strict",
      "--no-thinking",
    ];

    const child = spawn(candidate.command, args, {
      cwd: FINANCIAL_LOGIC_DIR,
      env: { ...process.env, ANTHROPIC_API_KEY: apiKey },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill();
      reject(
        new FinancialAnalysisError(
          "claude_analysis_failure",
          "Cost estimate (analyze.py --dry-run) timed out",
          504,
          { timeoutMs }
        )
      );
    }, timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      reject(
        new FinancialAnalysisError(
          "claude_analysis_failure",
          "Cost estimate (analyze.py --dry-run) could not be started",
          502,
          {
            startFailure: true,
            candidate: candidate.command,
            source: candidate.source,
            message: error instanceof Error ? error.message : String(error),
            code: isRecord(error) ? error.code : undefined,
          }
        )
      );
    });

    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);

      const estimate = parseDryRunEstimate(stdout, model);

      if (!estimate) {
        reject(
          new FinancialAnalysisError(
            "claude_analysis_failure",
            "Cost estimate (analyze.py --dry-run) produced no parseable token/cost output",
            502,
            {
              code,
              stdoutTail: stdout.slice(-2000),
              stderrTail: stderr.slice(-2000),
            }
          )
        );
        return;
      }

      resolve(estimate);
    });
  });
}

async function analyzeMarkdownWithClaude(
  documents: Array<ExtractedFinancialStatement & { markdown: string }>,
  apiKey: string,
  model: string
): Promise<ClaudeAnalysisResult> {
  const userPrompt = buildClaudeUserPrompt(documents);

  try {
    const first = await callClaudeMessages(apiKey, model, userPrompt);
    let json: JsonRecord | null = null;
    let validationErrors: string[] = [];

    try {
      json = parseClaudeJson(first.text);
    } catch (error) {
      validationErrors = [
        error instanceof Error ? error.message : String(error),
      ];
    }

    if (json) {
      const validation = await validateFinancialAnalysisJsonWithLogic(json);

      if (validation.isValid) {
        return { json, model, usage: first.usage };
      }

      validationErrors = validation.errors;
    }

    const correctionPrompt = buildClaudeCorrectionPrompt(
      userPrompt,
      first.text,
      validationErrors
    );
    const corrected = await callClaudeMessages(apiKey, model, correctionPrompt);
    const correctedJson = parseClaudeJson(corrected.text);
    const correctedValidation =
      await validateFinancialAnalysisJsonWithLogic(correctedJson);

    if (!correctedValidation.isValid) {
      throw new Error(correctedValidation.errors.join("; "));
    }

    return {
      json: correctedJson,
      model,
      usage: mergeClaudeUsage(first.usage, corrected.usage),
    };
  } catch (error) {
    if (error instanceof FinancialAnalysisError) {
      throw error;
    }

    throw new FinancialAnalysisError(
      "claude_analysis_failure",
      "Claude analysis failed to return valid financial analysis JSON",
      502,
      error instanceof Error ? error.message : String(error)
    );
  }
}

async function validateFinancialAnalysisJsonWithLogic(
  data: JsonRecord
): Promise<LogicRendererValidation> {
  try {
    const result = await runLogicRendererBridge("validate", data);
    return normalizeLogicValidation(result.validation);
  } catch (error) {
    if (!isRendererUnavailableError(error)) {
      throw error;
    }

    return validateFinancialAnalysisJsonLocally(data, [
      getRendererUnavailableWarning(error),
    ]);
  }
}

async function renderFinancialAnalysisWithLogic(data: JsonRecord): Promise<{
  html: string;
  validation: LogicRendererValidation;
}> {
  let result: LogicRendererResult;

  try {
    result = await runLogicRendererBridge("render", data);
  } catch (error) {
    if (!isRendererUnavailableError(error)) {
      throw error;
    }

    const validation = validateFinancialAnalysisJsonLocally(data, [
      getRendererUnavailableWarning(error),
    ]);

    if (!validation.isValid) {
      throw new FinancialAnalysisError(
        "malformed_intermediate_output",
        "Financial analysis JSON is not compatible with the dashboard HTML fallback",
        502,
        { errors: validation.errors, warnings: validation.warnings }
      );
    }

    return {
      html: renderFinancialAnalysisHtmlFallback(data, validation.warnings),
      validation,
    };
  }

  const validation = normalizeLogicValidation(result.validation);

  if (!validation.isValid) {
    throw new FinancialAnalysisError(
      "malformed_intermediate_output",
      "Financial analysis JSON is not compatible with the financial-statement-analysis-logic renderer",
      502,
      { errors: validation.errors, warnings: validation.warnings }
    );
  }

  if (typeof result.html !== "string" || !result.html.trim()) {
    throw new FinancialAnalysisError(
      "malformed_intermediate_output",
      "Financial statement logic renderer did not return report HTML",
      502,
      result
    );
  }

  return {
    html: result.html,
    validation,
  };
}

export async function exportFinancialAnalysisArtifact(input: {
  format: FinancialAnalysisExportFormat;
  report: unknown;
  fileName?: string;
}): Promise<FinancialAnalysisExportArtifact> {
  const normalized = normalizeFinancialAnalysisReportForExport(input.report);
  const baseName = slugifyFileName(input.fileName || getExportCompanyName(normalized.json));

  if (input.format === "html") {
    let result: LogicRendererResult;

    try {
      result = await runLogicRendererBridge("render", normalized.json);
    } catch (error) {
      if (!isRendererUnavailableError(error)) {
        throw error;
      }

      const validation = validateFinancialAnalysisJsonLocally(normalized.json, [
        getRendererUnavailableWarning(error),
      ]);

      if (!validation.isValid) {
        throw new FinancialAnalysisError(
          "malformed_intermediate_output",
          "Financial analysis JSON is not compatible with HTML export",
          400,
          { errors: validation.errors, warnings: validation.warnings }
        );
      }

      return {
        content: toArrayBuffer(
          new TextEncoder().encode(
            renderFinancialAnalysisHtmlFallback(normalized.json, validation.warnings)
          )
        ),
        contentType: "text/html; charset=utf-8",
        fileName: `${baseName || "financial-analysis"}.html`,
      };
    }

    if (typeof result.html !== "string" || !result.html.trim()) {
      throw new FinancialAnalysisError(
        "dashboard_execution_failure",
        "Financial statement logic renderer did not return report HTML",
        502,
        result
      );
    }

    return {
      content: toArrayBuffer(new TextEncoder().encode(result.html)),
      contentType: "text/html; charset=utf-8",
      fileName: `${baseName || "financial-analysis"}.html`,
    };
  }

  const mode = input.format === "pdf" ? "pdf" : "excel";
  const result = await runLogicRendererBridge(mode, normalized.json);
  const contentBase64 =
    typeof result.contentBase64 === "string" ? result.contentBase64 : "";

  if (!contentBase64) {
    throw new FinancialAnalysisError(
      "dashboard_execution_failure",
      "Financial statement export renderer did not return file content",
      502,
      result
    );
  }

  const extension =
    typeof result.fileExtension === "string" && result.fileExtension.trim()
      ? result.fileExtension.trim()
      : input.format === "pdf"
        ? "pdf"
        : "xlsx";
  const contentType =
    typeof result.contentType === "string" && result.contentType.trim()
      ? result.contentType.trim()
      : input.format === "pdf"
        ? "application/pdf"
        : "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

  return {
    content: toArrayBuffer(Buffer.from(contentBase64, "base64")),
    contentType,
    fileName: `${baseName || "financial-analysis"}.${extension}`,
  };
}

async function runLogicRendererBridge(
  mode: LogicRendererMode,
  data: JsonRecord
): Promise<LogicRendererResult> {
  const failures: JsonRecord[] = [];
  const configuredServiceCandidates = getRendererServiceCandidates(false);
  const pythonCandidates = getPythonRendererCandidates();
  const defaultServiceCandidates = getRendererServiceCandidates(true).filter(
    (candidate) =>
      !configuredServiceCandidates.some(
        (configured) => configured.baseUrl === candidate.baseUrl
      )
  );

  for (const candidate of configuredServiceCandidates) {
    try {
      return await runLogicRendererServiceCandidate(mode, data, candidate);
    } catch (error) {
      if (!isRendererUnavailableError(error)) {
        throw error;
      }

      failures.push(getRendererFailureDetail("service", candidate, error));
    }
  }

  try {
    return await runLogicRendererBridgeWithCandidates(
      mode,
      data,
      pythonCandidates
    );
  } catch (error) {
    if (!isRendererUnavailableError(error)) {
      throw error;
    }

    failures.push(getRendererFailureDetail("python", null, error));
  }

  for (const candidate of defaultServiceCandidates) {
    try {
      return await runLogicRendererServiceCandidate(mode, data, candidate);
    } catch (error) {
      if (!isRendererUnavailableError(error)) {
        throw error;
      }

      failures.push(getRendererFailureDetail("service", candidate, error));
    }
  }

  throw new FinancialAnalysisError(
    "dashboard_execution_failure",
    "Financial statement logic renderer could not be started",
    502,
    {
      configuredServiceCandidates,
      pythonCandidates: pythonCandidates.map((candidate) => ({
        command: candidate.command,
        source: candidate.source,
      })),
      defaultServiceCandidates,
      failures,
    }
  );
}

async function runLogicRendererBridgeWithCandidates(
  mode: LogicRendererMode,
  data: JsonRecord,
  candidates: PythonRendererCandidate[]
): Promise<LogicRendererResult> {
  if (!candidates.length) {
    throw new FinancialAnalysisError(
      "dashboard_execution_failure",
      "Financial statement logic renderer could not be started",
      502,
      { reason: "No Python renderer candidates were configured" }
    );
  }

  const failures: JsonRecord[] = [];

  for (const candidate of candidates) {
    try {
      return await runLogicRendererBridgeCandidate(mode, data, candidate);
    } catch (error) {
      if (
        error instanceof FinancialAnalysisError &&
        error.code === "dashboard_execution_failure" &&
        isRendererUnavailableError(error)
      ) {
        failures.push({
          candidate: candidate.command,
          source: candidate.source,
          detail: error.detail,
        });
        continue;
      }

      throw error;
    }
  }

  throw new FinancialAnalysisError(
    "dashboard_execution_failure",
    "Financial statement logic renderer could not be started",
    502,
    {
      logicDir: FINANCIAL_LOGIC_DIR,
      bridge: FINANCIAL_LOGIC_BRIDGE,
      candidates: candidates.map((candidate) => ({
        command: candidate.command,
        source: candidate.source,
      })),
      failures,
    }
  );
}

function runLogicRendererBridgeCandidate(
  mode: LogicRendererMode,
  data: JsonRecord,
  candidate: PythonRendererCandidate
): Promise<LogicRendererResult> {
  return new Promise((resolve, reject) => {
    const timeoutMs = getPositiveNumberEnv(
      "FINANCIAL_RENDERER_TIMEOUT_MS",
      DEFAULT_RENDERER_TIMEOUT_MS
    );

    logFinancialStage({
      stage: "renderer",
      extra: {
        mode,
        logicDir: FINANCIAL_LOGIC_DIR,
        bridge: FINANCIAL_LOGIC_BRIDGE,
        pythonCandidate: candidate.command,
        pythonCandidateSource: candidate.source,
      },
    });

    const child = spawn(
      candidate.command,
      [...candidate.argsPrefix, FINANCIAL_LOGIC_BRIDGE],
      {
        cwd: FINANCIAL_LOGIC_DIR,
        env: process.env,
        stdio: ["pipe", "pipe", "pipe"],
      }
    );
    let stdout = "";
    let stderr = "";
    let settled = false;

    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill();
      reject(
        new FinancialAnalysisError(
          "dashboard_execution_failure",
          "Financial statement logic renderer timed out",
          504,
          { mode, timeoutMs }
        )
      );
    }, timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");

    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });

    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      reject(
        new FinancialAnalysisError(
          "dashboard_execution_failure",
          "Financial statement logic renderer could not be started",
          502,
          {
            startFailure: true,
            candidate: candidate.command,
            source: candidate.source,
            message: error instanceof Error ? error.message : String(error),
            code: isRecord(error) ? error.code : undefined,
          }
        )
      );
    });

    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);

      let parsed: unknown;

      try {
        parsed = JSON.parse(stdout);
      } catch (error) {
        reject(
          new FinancialAnalysisError(
            "dashboard_execution_failure",
            "Financial statement logic renderer returned malformed output",
            502,
            {
              code,
              stdout,
              stderr,
              parseError: error instanceof Error ? error.message : String(error),
            }
          )
        );
        return;
      }

      if (!isRecord(parsed)) {
        reject(
          new FinancialAnalysisError(
            "dashboard_execution_failure",
            "Financial statement logic renderer returned an invalid response",
            502,
            { code, parsed, stderr }
          )
        );
        return;
      }

      const result = parsed as LogicRendererResult;

      if (code !== 0 || result.success !== true) {
        reject(
          new FinancialAnalysisError(
            "dashboard_execution_failure",
            "Financial statement logic renderer failed",
            502,
            { code, error: result.error, traceback: result.traceback, stderr }
          )
        );
        return;
      }

      resolve(result);
    });

    child.stdin.end(JSON.stringify({ mode, data }));
  });
}

async function runLogicRendererServiceCandidate(
  mode: LogicRendererMode,
  data: JsonRecord,
  candidate: RendererServiceCandidate
): Promise<LogicRendererResult> {
  const timeoutMs = getPositiveNumberEnv(
    "FINANCIAL_RENDERER_TIMEOUT_MS",
    DEFAULT_RENDERER_TIMEOUT_MS
  );
  const endpoint = getRendererBridgeEndpoint(candidate.baseUrl);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  logFinancialStage({
    stage: "renderer",
    extra: {
      mode,
      rendererServiceUrl: candidate.baseUrl,
      rendererServiceSource: candidate.source,
    },
  });

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ mode, data }),
      signal: controller.signal,
    });
    const responseBody = await readResponseBody(response);

    if (!response.ok) {
      if (
        (response.status === 404 || response.status === 405) &&
        (mode === "render" || mode === "validate")
      ) {
        return await runLegacyRendererServiceAnalyzeCandidate(
          mode,
          data,
          candidate
        );
      }

      throw new FinancialAnalysisError(
        "dashboard_execution_failure",
        "Financial statement logic renderer service failed",
        response.status >= 400 && response.status < 600 ? response.status : 502,
        {
          serviceFailure: true,
          endpoint,
          source: candidate.source,
          status: response.status,
          body: responseBody,
        }
      );
    }

    if (!isRecord(responseBody)) {
      if (mode === "render" || mode === "validate") {
        return await runLegacyRendererServiceAnalyzeCandidate(
          mode,
          data,
          candidate
        );
      }

      throw new FinancialAnalysisError(
        "dashboard_execution_failure",
        "Financial statement logic renderer service returned an invalid response",
        502,
        {
          serviceFailure: true,
          endpoint,
          source: candidate.source,
          body: responseBody,
        }
      );
    }

    const result = responseBody as LogicRendererResult;

    if (result.success !== true) {
      throw new FinancialAnalysisError(
        "dashboard_execution_failure",
        "Financial statement logic renderer service failed",
        502,
        {
          serviceFailure: true,
          endpoint,
          source: candidate.source,
          error: result.error,
          traceback: result.traceback,
        }
      );
    }

    return result;
  } catch (error) {
    if (error instanceof FinancialAnalysisError) {
      throw error;
    }

    throw new FinancialAnalysisError(
      "dashboard_execution_failure",
      "Financial statement logic renderer service could not be reached",
      502,
      {
        serviceFailure: true,
        endpoint,
        source: candidate.source,
        message: error instanceof Error ? error.message : String(error),
      }
    );
  } finally {
    clearTimeout(timeout);
  }
}

async function runLegacyRendererServiceAnalyzeCandidate(
  mode: Extract<LogicRendererMode, "validate" | "render">,
  data: JsonRecord,
  candidate: RendererServiceCandidate
): Promise<LogicRendererResult> {
  const timeoutMs = getPositiveNumberEnv(
    "FINANCIAL_RENDERER_TIMEOUT_MS",
    DEFAULT_RENDERER_TIMEOUT_MS
  );
  const endpoint = getRendererAnalyzeEndpoint(candidate.baseUrl);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  const formData = new FormData();

  formData.append(
    "file",
    new Blob([JSON.stringify(data)], { type: "application/json" }),
    "financial-analysis.json"
  );

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    const responseBody = await readResponseBody(response);

    if (!response.ok) {
      const validationFailure = extractRendererValidationFailure(responseBody);

      if (validationFailure) {
        return {
          success: true,
          validation: validationFailure,
          html: null,
        };
      }

      throw new FinancialAnalysisError(
        "dashboard_execution_failure",
        "Financial statement legacy renderer service failed",
        response.status >= 400 && response.status < 600 ? response.status : 502,
        {
          serviceFailure: true,
          endpoint,
          source: candidate.source,
          status: response.status,
          body: responseBody,
        }
      );
    }

    if (!isRecord(responseBody) || responseBody.success !== true) {
      throw new FinancialAnalysisError(
        "dashboard_execution_failure",
        "Financial statement legacy renderer service returned an invalid response",
        502,
        {
          serviceFailure: true,
          endpoint,
          source: candidate.source,
          body: responseBody,
        }
      );
    }

    const warnings = isStringArray(responseBody.warnings)
      ? responseBody.warnings
      : [];
    const validation = {
      isValid: true,
      errors: [],
      warnings,
    };

    if (mode === "validate") {
      return { success: true, validation };
    }

    if (typeof responseBody.html !== "string" || !responseBody.html.trim()) {
      throw new FinancialAnalysisError(
        "dashboard_execution_failure",
        "Financial statement legacy renderer service did not return report HTML",
        502,
        {
          serviceFailure: true,
          endpoint,
          source: candidate.source,
          body: responseBody,
        }
      );
    }

    return {
      success: true,
      validation,
      html: responseBody.html,
    };
  } catch (error) {
    if (error instanceof FinancialAnalysisError) {
      throw error;
    }

    throw new FinancialAnalysisError(
      "dashboard_execution_failure",
      "Financial statement legacy renderer service could not be reached",
      502,
      {
        serviceFailure: true,
        endpoint,
        source: candidate.source,
        message: error instanceof Error ? error.message : String(error),
      }
    );
  } finally {
    clearTimeout(timeout);
  }
}

function getRendererBridgeEndpoint(baseUrl: string) {
  const normalizedBaseUrl = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;

  return new URL("render-bridge", normalizedBaseUrl).toString();
}

function getRendererAnalyzeEndpoint(baseUrl: string) {
  const normalizedBaseUrl = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;

  return new URL("analyze", normalizedBaseUrl).toString();
}

function getRendererServiceCandidates(includeDefault: boolean) {
  const candidates: RendererServiceCandidate[] = [];

  addRendererServiceCandidate(
    candidates,
    process.env.FINANCIAL_RENDERER_API_URL,
    "FINANCIAL_RENDERER_API_URL"
  );
  addRendererServiceCandidate(
    candidates,
    process.env.FINANCIAL_STATEMENT_ANALYSIS_API_URL,
    "FINANCIAL_STATEMENT_ANALYSIS_API_URL"
  );
  addRendererServiceCandidate(
    candidates,
    process.env.FINANCIAL_ANALYSIS_API_URL,
    "FINANCIAL_ANALYSIS_API_URL"
  );

  if (
    includeDefault &&
    process.env.FINANCIAL_RENDERER_DISABLE_DEFAULT_API !== "1"
  ) {
    addRendererServiceCandidate(
      candidates,
      DEFAULT_FINANCIAL_RENDERER_API_URL,
      "default-kredit-lab-renderer-api"
    );
  }

  return candidates;
}

function addRendererServiceCandidate(
  candidates: RendererServiceCandidate[],
  baseUrl: string | undefined,
  source: string
) {
  const trimmedBaseUrl = normalizeRendererServiceBaseUrl(baseUrl);

  if (!trimmedBaseUrl) return;
  if (candidates.some((candidate) => candidate.baseUrl === trimmedBaseUrl)) {
    return;
  }

  candidates.push({ baseUrl: trimmedBaseUrl, source });
}

function normalizeRendererServiceBaseUrl(baseUrl: string | undefined) {
  const trimmed = baseUrl?.trim().replace(/\/+$/, "");

  if (!trimmed) return "";

  return /^[a-z][a-z\d+\-.]*:\/\//i.test(trimmed)
    ? trimmed
    : `https://${trimmed}`;
}

function getPythonRendererCandidates(): PythonRendererCandidate[] {
  const candidates: PythonRendererCandidate[] = [];

  addPythonRendererCandidate(
    candidates,
    process.env.FINANCIAL_RENDERER_PYTHON_BIN,
    "FINANCIAL_RENDERER_PYTHON_BIN"
  );
  addPythonRendererCandidate(candidates, process.env.PYTHON, "PYTHON");

  if (process.platform === "win32") {
    addPythonRendererCandidate(candidates, "py", "windows-py-launcher", ["-3"]);
    addWindowsPythonInstallCandidates(candidates);
    addPythonRendererCandidate(candidates, "python", "windows-python");
    addPythonRendererCandidate(candidates, "python3", "windows-python3");
  } else {
    addUnixPythonInstallCandidates(candidates);
    addPythonRendererCandidate(candidates, "python3.11", "python3.11");
    addPythonRendererCandidate(candidates, "python3", "python3");
    addPythonRendererCandidate(candidates, "python", "python");
  }

  return candidates;
}

function addWindowsPythonInstallCandidates(candidates: PythonRendererCandidate[]) {
  const localAppData = process.env.LOCALAPPDATA;
  const programFiles = process.env.ProgramFiles;
  const programFilesX86 = process.env["ProgramFiles(x86)"];
  const userProfile = process.env.USERPROFILE;
  const versions = ["Python311", "Python312", "Python310"];

  if (userProfile) {
    addPythonRendererCandidate(
      candidates,
      path.join(
        userProfile,
        ".cache",
        "codex-runtimes",
        "codex-primary-runtime",
        "dependencies",
        "python",
        "python.exe"
      ),
      "codex-primary-runtime-python"
    );
  }

  for (const version of versions) {
    if (localAppData) {
      addPythonRendererCandidate(
        candidates,
        path.join(localAppData, "Programs", "Python", version, "python.exe"),
        `LOCALAPPDATA-${version}`
      );
    }

    if (programFiles) {
      addPythonRendererCandidate(
        candidates,
        path.join(programFiles, "Python", version, "python.exe"),
        `ProgramFiles-${version}`
      );
      addPythonRendererCandidate(
        candidates,
        path.join(programFiles, version, "python.exe"),
        `ProgramFilesDirect-${version}`
      );
    }

    if (programFilesX86) {
      addPythonRendererCandidate(
        candidates,
        path.join(programFilesX86, "Python", version, "python.exe"),
        `ProgramFilesX86-${version}`
      );
      addPythonRendererCandidate(
        candidates,
        path.join(programFilesX86, version, "python.exe"),
        `ProgramFilesX86Direct-${version}`
      );
    }
  }
}

function addUnixPythonInstallCandidates(candidates: PythonRendererCandidate[]) {
  for (const command of [
    "/opt/homebrew/bin/python3.11",
    "/usr/local/bin/python3.11",
    "/usr/bin/python3.11",
    "/usr/bin/python3",
  ]) {
    addPythonRendererCandidate(candidates, command, "absolute-python-path");
  }
}

function addPythonRendererCandidate(
  candidates: PythonRendererCandidate[],
  command: string | undefined,
  source: string,
  argsPrefix: string[] = []
) {
  const trimmedCommand = command?.trim();

  if (!trimmedCommand) return;

  const alreadyAdded = candidates.some(
    (candidate) =>
      candidate.command === trimmedCommand &&
      candidate.argsPrefix.join("\u0000") === argsPrefix.join("\u0000")
  );

  if (alreadyAdded) return;

  candidates.push({ command: trimmedCommand, argsPrefix, source });
}

function isRendererStartFailure(detail: unknown) {
  return isRecord(detail) && detail.startFailure === true;
}

function isRendererUnavailableError(error: unknown) {
  if (
    !(error instanceof FinancialAnalysisError) ||
    error.code !== "dashboard_execution_failure"
  ) {
    return false;
  }

  const detail = error.detail;

  if (isRendererStartFailure(detail)) return true;

  if (isRecord(detail) && detail.serviceFailure === true) {
    const status = typeof detail.status === "number" ? detail.status : 0;

    if (!status || status === 404 || status === 405 || status >= 500) {
      return true;
    }
  }

  const message = [
    error.message,
    isRecord(detail) && typeof detail.error === "string" ? detail.error : "",
    isRecord(detail) && typeof detail.stderr === "string" ? detail.stderr : "",
    isRecord(detail) && typeof detail.traceback === "string"
      ? detail.traceback
      : "",
    isRecord(detail) && typeof detail.message === "string" ? detail.message : "",
  ]
    .join("\n")
    .toLowerCase();

  return (
    message.includes("no module named") ||
    message.includes("cannot find module") ||
    message.includes("could not be started") ||
    message.includes("could not be reached") ||
    message.includes("enoent") ||
    message.includes("eacces") ||
    message.includes("permission denied") ||
    message.includes("failed to fetch") ||
    message.includes("fetch failed") ||
    message.includes("connection refused") ||
    message.includes("timed out") ||
    message.includes("aborted")
  );
}

function getRendererFailureDetail(
  kind: "python" | "service",
  candidate: PythonRendererCandidate | RendererServiceCandidate | null,
  error: unknown
): JsonRecord {
  if (error instanceof FinancialAnalysisError) {
    return {
      kind,
      candidate,
      message: error.message,
      status: error.status,
      detail: error.detail,
    };
  }

  return {
    kind,
    candidate,
    message: error instanceof Error ? error.message : String(error),
  };
}

function extractRendererValidationFailure(
  responseBody: unknown
): LogicRendererValidation | null {
  if (!isRecord(responseBody)) return null;

  const detail = asRecord(responseBody.detail);
  const detailCode = getStringValue(detail.code);

  if (
    detailCode !== "renderer_validation_failure" &&
    detailCode !== "malformed_intermediate_output"
  ) {
    return null;
  }

  const nestedDetail = asRecord(detail.detail);
  const errors = isStringArray(nestedDetail.errors)
    ? nestedDetail.errors
    : [getStringValue(detail.message) || "Renderer validation failed"];
  const warnings = isStringArray(nestedDetail.warnings)
    ? nestedDetail.warnings
    : [];

  return {
    isValid: false,
    errors,
    warnings,
  };
}

function getRendererUnavailableWarning(error: unknown) {
  const detail =
    error instanceof FinancialAnalysisError
      ? summarizeRendererFailureDetail(error.detail)
      : "";
  const suffix = detail ? ` Detail: ${detail}` : "";

  return `Official financial statement renderer was unavailable, so the dashboard used its local HTML fallback.${suffix}`;
}

function summarizeRendererFailureDetail(detail: unknown): string {
  if (!isRecord(detail)) return "";

  const failures = detail.failures;

  if (Array.isArray(failures) && failures.length > 0) {
    const messages = failures
      .map((failure) => {
        if (!isRecord(failure)) return "";
        const failureDetail = failure.detail;

        if (isRecord(failureDetail)) {
          if (typeof failureDetail.message === "string") {
            return failureDetail.message;
          }

          if (typeof failureDetail.error === "string") {
            return failureDetail.error;
          }
        }

        if (typeof failure.message === "string") return failure.message;

        return "";
      })
      .filter(Boolean);

    return [...new Set(messages)].slice(0, 3).join("; ");
  }

  if (typeof detail.message === "string") return detail.message;
  if (typeof detail.error === "string") return detail.error;

  return "";
}

function normalizeLogicValidation(value: unknown): LogicRendererValidation {
  if (!isRecord(value)) {
    throw new FinancialAnalysisError(
      "malformed_intermediate_output",
      "Financial statement logic validation output is malformed",
      502,
      value
    );
  }

  const errors = value.errors;
  const warnings = value.warnings;

  if (
    typeof value.isValid !== "boolean" ||
    !isStringArray(errors) ||
    !isStringArray(warnings)
  ) {
    throw new FinancialAnalysisError(
      "malformed_intermediate_output",
      "Financial statement logic validation output is malformed",
      502,
      value
    );
  }

  return {
    isValid: value.isValid,
    errors,
    warnings,
  };
}

async function callClaudeMessages(
  apiKey: string,
  model: string,
  userPrompt: string
): Promise<ClaudeMessagesResult> {
  const maxTokens = getClaudeMaxOutputTokens(model);
  const effort = resolveClaudeEffort(model, process.env.ANTHROPIC_EFFORT);
  const requestBody = {
    model,
    max_tokens: maxTokens,
    system: CLAUDE_SYSTEM_PROMPT,
    messages: [
      {
        role: "user",
        content: userPrompt,
      },
    ],
    ...(effort ? { output_config: { effort } } : {}),
  };
  const costEstimate = estimateClaudeCost({
    modelId: model,
    inputCharacters: userPrompt.length,
    outputTokens: maxTokens,
  });

  logFinancialStage({
    stage: "claude_analysis",
    claudeInputTextLength: userPrompt.length,
    claudeModel: model,
    extra: {
      requestParameterKeys: Object.keys(requestBody),
      claudeEffort: effort,
      estimatedInputTokens: costEstimate.inputTokens,
      estimatedOutputTokens: costEstimate.outputTokens,
      estimatedCostUsd: costEstimate.totalCostUsd,
    },
  });

  const response = await fetch(ANTHROPIC_MESSAGES_URL, {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": ANTHROPIC_VERSION,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });
  const responseBody = await readResponseBody(response);

  if (!response.ok) {
    const errorCode = getClaudeErrorCode(response.status, responseBody);
    const message = getClaudeErrorMessage(errorCode, responseBody);

    logFinancialStage({
      stage: "claude_analysis",
      claudeInputTextLength: userPrompt.length,
      claudeModel: model,
      error: responseBody,
      extra: {
        status: response.status,
        requestParameterKeys: Object.keys(requestBody),
      },
    });

    throw new FinancialAnalysisError(
      errorCode,
      message,
      getClaudeHttpStatus(errorCode, response.status),
      { status: response.status, body: responseBody }
    );
  }

  const text = extractClaudeText(responseBody);
  const stopReason = isRecord(responseBody)
    ? getStringValue(responseBody.stop_reason)
    : "";
  const stopSequence = isRecord(responseBody)
    ? getStringValue(responseBody.stop_sequence)
    : "";
  const usage = isRecord(responseBody) ? responseBody.usage : undefined;

  logFinancialStage({
    stage: "claude_analysis",
    claudeInputTextLength: userPrompt.length,
    claudeModel: model,
    extra: {
      stopReason,
      stopSequence,
      responseTextLength: text.length,
      maxTokens,
      usage: isRecord(usage) ? usage : undefined,
    },
  });

  if (stopReason === "max_tokens") {
    throwClaudeOutputTruncated({
      model,
      maxTokens,
      usage,
      responseTextLength: text.length,
    });
  }

  if (!text.trim()) {
    throw new FinancialAnalysisError(
      "claude_analysis_failure",
      "Claude analysis returned an empty response",
      502,
      responseBody
    );
  }

  return {
    text,
    usage,
    stopReason,
    stopSequence,
  };
}

function throwClaudeOutputTruncated(input: {
  model: string;
  maxTokens: number;
  usage?: unknown;
  responseTextLength: number;
}): never {
  throw new FinancialAnalysisError(
    "claude_output_truncated",
    `Claude hit the ${input.maxTokens.toLocaleString()} output-token limit before completing renderer JSON. No correction request was sent to avoid additional token spend. Increase ANTHROPIC_MAX_TOKENS for a model that supports it, choose a higher-output Claude model, or reduce the selected input files.`,
    502,
    {
      model: input.model,
      maxTokens: input.maxTokens,
      usage: input.usage,
      responseTextLength: input.responseTextLength,
      stopReason: "max_tokens",
    }
  );
}

function mergeClaudeUsage(...usages: unknown[]) {
  const merged: JsonRecord = {};

  for (const usage of usages) {
    if (!isRecord(usage)) continue;

    for (const [key, value] of Object.entries(usage)) {
      if (typeof value === "number") {
        merged[key] =
          (typeof merged[key] === "number" ? merged[key] : 0) + value;
      } else if (!(key in merged)) {
        merged[key] = value;
      }
    }
  }

  return Object.keys(merged).length > 0 ? merged : undefined;
}

function getClaudeMaxOutputTokens(modelId: string) {
  const configuredMaxTokens = getPositiveNumberEnv(
    "ANTHROPIC_MAX_TOKENS",
    DEFAULT_ANTHROPIC_MAX_TOKENS
  );
  const modelMaxTokens =
    getClaudeModelById(modelId)?.maxOutputTokens || DEFAULT_ANTHROPIC_MAX_TOKENS;

  return Math.min(configuredMaxTokens, modelMaxTokens);
}

function buildClaudeUserPrompt(
  documents: Array<ExtractedFinancialStatement & { markdown: string }>
) {
  const files = documents
    .map(
      (document, index) => `
===== SOURCE DOCUMENT ${index + 1}: ${document.fileName} (${document.sourceKind}) =====
${document.markdown}
`.trim()
    )
    .join("\n\n");

  return `
Analyze the following financial statement source documents and return the Kredit
Lab renderer-compatible JSON only.

${files}
`.trim();
}

function buildClaudeCorrectionPrompt(
  originalPrompt: string,
  previousResponse: string,
  validationErrors: string[]
) {
  return `
The previous response was not valid renderer-compatible JSON.

Validation errors:
${validationErrors.map((error) => `- ${error}`).join("\n")}

Previous response:
${previousResponse}

Original task:
${originalPrompt}

Return corrected JSON only.
`.trim();
}

function parseClaudeJson(text: string): JsonRecord {
  const trimmed = text.trim();
  const candidates = [
    trimmed,
    trimmed.replace(/^```(?:json)?/i, "").replace(/```$/i, "").trim(),
    extractFirstJsonObject(trimmed),
  ].filter((candidate): candidate is string => Boolean(candidate));

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);

      if (isRecord(parsed)) {
        return parsed;
      }
    } catch {
      // Try the next candidate.
    }
  }

  throw new Error("Claude response was not a JSON object");
}

function extractFirstJsonObject(text: string) {
  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");

  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    return null;
  }

  return text.slice(firstBrace, lastBrace + 1);
}

function validateExtractedStatements(statements: ExtractedFinancialStatement[]) {
  if (!statements.length) {
    throw new FinancialAnalysisError(
      "malformed_intermediate_output",
      "Financial statement extraction returned no documents",
      502
    );
  }

  for (const statement of statements) {
    if (
      statement.markdown !== undefined &&
      typeof statement.markdown !== "string"
    ) {
      throw new FinancialAnalysisError(
        "malformed_intermediate_output",
        "Extractor markdown output is malformed",
        502,
        { fileName: statement.fileName }
      );
    }

    if (
      statement.analysisJson !== undefined &&
      !isRecord(statement.analysisJson)
    ) {
      throw new FinancialAnalysisError(
        "malformed_intermediate_output",
        "Extractor JSON output is malformed",
        502,
        { fileName: statement.fileName }
      );
    }
  }
}

const LOCAL_FALLBACK_REQUIRED_SECTIONS = [
  "company_info",
  "statement_of_comprehensive_income",
  "statement_of_financial_position",
  "financial_ratios",
  "working_capital_analysis",
  "funding_mismatch_analysis",
  "funding_profile",
  "tnw_analysis",
  "dscr_analysis",
  "integrity_check",
  "analysis_summary",
  "report_footer",
];

function validateFinancialAnalysisJsonLocally(
  data: JsonRecord,
  extraWarnings: string[] = []
): LogicRendererValidation {
  const errors: string[] = [];
  const warnings = [...extraWarnings];
  const company = getCompanyInfo(data);
  const schemaInfo = asRecord(data._schema_info);
  const version = getStringValue(schemaInfo.version);

  if (!version.startsWith("v7") && !version.startsWith("v6")) {
    warnings.push("Missing or unexpected _schema_info.version");
  }

  for (const section of LOCAL_FALLBACK_REQUIRED_SECTIONS) {
    if (!(section in data)) {
      errors.push(`Missing required section: ${section}`);
    }
  }

  if (!getStringValue(company.legal_name) && !getStringValue(company.name)) {
    warnings.push("Missing company name");
  }

  if (getFallbackPeriodKeys(data).length === 0) {
    errors.push("No valid financial periods found");
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

function renderFinancialAnalysisHtmlFallback(
  data: JsonRecord,
  warnings: string[]
) {
  const company = getCompanyInfo(data);
  const companyName =
    getStringValue(company.legal_name) ||
    getStringValue(company.name) ||
    "Financial Statement Analysis";
  const schemaInfo = asRecord(data._schema_info);
  const currencyUnit = getStringValue(schemaInfo.currency_unit) || "RM";
  const periods = getFallbackPeriodKeys(data);

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${escapeHtml(companyName)} - Financial Statement Analysis</title>
${renderFallbackCss()}
</head>
<body>
<main class="kl-report">
  <header class="kl-header">
    <div>
      <p class="kl-eyebrow">Kredit Lab Financial Statement Analysis</p>
      <h1>${escapeHtml(companyName)}</h1>
      <p>${escapeHtml(getStringValue(company.principal_activities) || "Credit-focused financial review")}</p>
    </div>
    <dl>
      ${renderFallbackDefinition("Currency", currencyUnit)}
      ${renderFallbackDefinition("Financial year end", getStringValue(company.financial_year_end) || "-")}
      ${renderFallbackDefinition("Generated", new Date().toISOString().slice(0, 10))}
    </dl>
  </header>
  ${renderFallbackWarningSection(warnings)}
  ${renderFallbackCompanySection(data, periods)}
  ${renderFallbackDataSection("Statement of Comprehensive Income", data.statement_of_comprehensive_income, periods, data)}
  ${renderFallbackDataSection("Statement of Financial Position", data.statement_of_financial_position, periods, data)}
  ${renderFallbackDataSection("Financial Ratios", data.financial_ratios, periods, data)}
  ${renderFallbackDataSection("Working Capital Analysis", data.working_capital_analysis, periods, data)}
  ${renderFallbackDataSection("Funding Mismatch Analysis", data.funding_mismatch_analysis, periods, data)}
  ${renderFallbackDataSection("Funding Profile", data.funding_profile, periods, data)}
  ${renderFallbackDataSection("TNW Analysis", data.tnw_analysis, periods, data)}
  ${renderFallbackDataSection("DSCR Analysis", data.dscr_analysis, periods, data)}
  ${renderFallbackDataSection("Integrity Check", data.integrity_check, periods, data)}
  ${renderFallbackDataSection("Analysis Summary", data.analysis_summary, periods, data)}
  ${renderFallbackFooter(data)}
</main>
</body>
</html>`;
}

function renderFallbackCss() {
  return `<style>
:root { color-scheme: light; --ink:#0f172a; --muted:#475569; --line:#dbe3ea; --panel:#fff; --soft:#f8fafc; --accent:#0891b2; --warn:#b45309; }
* { box-sizing: border-box; }
body { margin:0; background:#eef2f7; color:var(--ink); font-family:Arial, Helvetica, sans-serif; }
.kl-report { width:min(1180px, calc(100% - 32px)); margin:24px auto; }
.kl-header { display:grid; grid-template-columns:minmax(0, 1fr) auto; gap:24px; padding:28px; border:1px solid var(--line); background:var(--panel); }
.kl-header h1 { margin:4px 0 8px; font-size:30px; line-height:1.15; }
.kl-header p, .kl-header dl { margin:0; color:var(--muted); }
.kl-header dl { display:grid; min-width:260px; gap:10px; padding:16px; background:var(--soft); border:1px solid var(--line); }
.kl-header dt { font-size:11px; text-transform:uppercase; color:#64748b; }
.kl-header dd { margin:2px 0 0; font-weight:700; color:var(--ink); }
.kl-eyebrow { color:var(--accent) !important; font-size:12px; font-weight:700; text-transform:uppercase; }
.kl-section { margin-top:16px; border:1px solid var(--line); background:var(--panel); }
.kl-section-header { padding:18px 20px; border-bottom:1px solid var(--line); background:var(--soft); }
.kl-section-header h2 { margin:0; font-size:18px; }
.kl-section-body { padding:18px 20px; }
.kl-warning { border-color:#fbbf24; background:#fffbeb; color:#92400e; }
.kl-table-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
th { background:#f1f5f9; color:#334155; text-align:left; font-size:12px; text-transform:uppercase; }
td.number, th.number { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
.kl-group td { background:#e0f2fe; color:#0c4a6e; font-weight:700; }
.kl-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }
.kl-card { padding:14px; border:1px solid var(--line); background:var(--soft); }
.kl-card dt { color:var(--muted); font-size:12px; }
.kl-card dd { margin:4px 0 0; font-weight:700; }
.kl-muted { color:var(--muted); }
.kl-list { margin:0; padding-left:18px; }
.kl-list li { margin:8px 0; }
pre.kl-json { white-space:pre-wrap; overflow-x:auto; margin:0; font-size:12px; color:#334155; }
@media (max-width:760px) { .kl-report { width:calc(100% - 16px); margin:8px auto; } .kl-header { grid-template-columns:1fr; padding:20px; } .kl-header dl { min-width:0; } }
</style>`;
}

function renderFallbackCompanySection(data: JsonRecord, periods: string[]) {
  const company = getCompanyInfo(data);

  return renderFallbackSection(
    "Company Information",
    `<div class="kl-grid">
      ${renderFallbackInfoCard("Legal name", getStringValue(company.legal_name) || getStringValue(company.name) || "-")}
      ${renderFallbackInfoCard("Registration no.", getStringValue(company.registration_no) || "-")}
      ${renderFallbackInfoCard("Financial year end", getStringValue(company.financial_year_end) || "-")}
      ${renderFallbackInfoCard("Periods", periods.map((period) => getFallbackPeriodLabel(data, period)).join(", ") || "-")}
    </div>`
  );
}

function renderFallbackDataSection(
  title: string,
  section: unknown,
  periods: string[],
  data: JsonRecord
) {
  if (!isRecord(section) || Object.keys(section).length === 0) return "";

  const rows = renderFallbackRows(section, periods);

  if (!rows) return "";

  return renderFallbackSection(
    title,
    `<div class="kl-table-wrap"><table><thead><tr><th>Item</th>${periods
      .map((period) => `<th class="number">${escapeHtml(getFallbackPeriodLabel(data, period))}</th>`)
      .join("")}<th>Details</th></tr></thead><tbody>${rows}</tbody></table></div>`
  );
}

function renderFallbackRows(
  value: JsonRecord,
  periods: string[],
  depth = 0
): string {
  let rows = "";

  for (const [key, child] of Object.entries(value)) {
    if (key.startsWith("_")) continue;

    if (isRecord(child) && hasFallbackValues(child)) {
      rows += renderFallbackValuesRow(key, child, periods, depth);
      continue;
    }

    if (isRecord(child)) {
      if (hasFallbackScalarShape(child)) {
        rows += renderFallbackScalarRow(key, child, periods, depth);
      } else {
        rows += `<tr class="kl-group"><td colspan="${periods.length + 2}" style="padding-left:${depth * 16 + 12}px">${escapeHtml(titleize(key))}</td></tr>`;
        rows += renderFallbackRows(child, periods, depth + 1);
      }
      continue;
    }

    rows += renderFallbackPlainRow(key, child, periods, depth);
  }

  return rows;
}

function renderFallbackValuesRow(
  key: string,
  item: JsonRecord,
  periods: string[],
  depth: number
) {
  const values = asRecord(item.values || item.values_standard);
  const label = getStringValue(item.display_name) || titleize(key);
  const details = [
    getStringValue(item.formula),
    getStringValue(item.benchmark),
    getStringValue(item.unit),
  ].filter(Boolean);

  return `<tr><td style="padding-left:${depth * 16 + 12}px">${escapeHtml(label)}</td>${periods
    .map(
      (period) =>
        `<td class="number">${escapeHtml(formatFallbackScalar(values[period]))}</td>`
    )
    .join("")}<td class="kl-muted">${escapeHtml(details.join(" | "))}</td></tr>`;
}

function renderFallbackScalarRow(
  key: string,
  item: JsonRecord,
  periods: string[],
  depth: number
) {
  const label = getStringValue(item.display_name) || titleize(key);
  const value = item.value ?? item.amount ?? item.total ?? item.standard ?? item.current;

  return `<tr><td style="padding-left:${depth * 16 + 12}px">${escapeHtml(label)}</td>${periods
    .map(() => `<td class="number">-</td>`)
    .join("")}<td>${renderFallbackInlineValue(value)}</td></tr>`;
}

function renderFallbackPlainRow(
  key: string,
  value: unknown,
  periods: string[],
  depth: number
) {
  return `<tr><td style="padding-left:${depth * 16 + 12}px">${escapeHtml(titleize(key))}</td>${periods
    .map(() => `<td class="number">-</td>`)
    .join("")}<td>${renderFallbackInlineValue(value)}</td></tr>`;
}

function renderFallbackWarningSection(warnings: string[]) {
  if (!warnings.length) return "";

  return `<section class="kl-section kl-warning"><div class="kl-section-body"><ul class="kl-list">${warnings
    .map((warning) => `<li>${escapeHtml(warning)}</li>`)
    .join("")}</ul></div></section>`;
}

function renderFallbackFooter(data: JsonRecord) {
  const footer = asRecord(data.report_footer);
  const preparedBy = getStringValue(footer.prepared_by) || "Kredit Lab";
  const disclaimer =
    getStringValue(footer.disclaimer) ||
    "This report is generated from uploaded financial statement data and should be reviewed before credit decisions are made.";

  return `<footer class="kl-section"><div class="kl-section-body"><p><strong>Prepared by: ${escapeHtml(preparedBy)}</strong></p><p class="kl-muted">${escapeHtml(disclaimer)}</p></div></footer>`;
}

function renderFallbackSection(title: string, body: string) {
  return `<section class="kl-section"><div class="kl-section-header"><h2>${escapeHtml(title)}</h2></div><div class="kl-section-body">${body}</div></section>`;
}

function renderFallbackInfoCard(label: string, value: string) {
  return `<dl class="kl-card"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></dl>`;
}

function renderFallbackDefinition(label: string, value: string) {
  return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`;
}

function renderFallbackInlineValue(value: unknown): string {
  if (Array.isArray(value)) {
    if (!value.length) return "-";

    return `<ul class="kl-list">${value
      .map((item) => `<li>${renderFallbackInlineValue(item)}</li>`)
      .join("")}</ul>`;
  }

  if (isRecord(value)) {
    return `<pre class="kl-json">${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
  }

  return escapeHtml(formatFallbackScalar(value));
}

function getFallbackPeriodKeys(data: JsonRecord) {
  const company = getCompanyInfo(data);
  const periods = asRecord(company.periods_analyzed);

  if (Object.keys(periods).length > 0) {
    return Object.keys(periods);
  }

  return findFallbackPeriodKeys(data);
}

function findFallbackPeriodKeys(value: unknown): string[] {
  if (!isRecord(value)) return [];

  if (isRecord(value.values)) {
    return Object.keys(value.values);
  }

  for (const child of Object.values(value)) {
    const keys = findFallbackPeriodKeys(child);

    if (keys.length > 0) return keys;
  }

  return [];
}

function getFallbackPeriodLabel(data: JsonRecord, period: string) {
  const periods = asRecord(getCompanyInfo(data).periods_analyzed);
  const label = periods[period];

  return typeof label === "string" && label.trim() ? label : titleize(period);
}

function hasFallbackValues(value: JsonRecord) {
  return isRecord(value.values) || isRecord(value.values_standard);
}

function hasFallbackScalarShape(value: JsonRecord) {
  return (
    "value" in value ||
    "amount" in value ||
    "total" in value ||
    "standard" in value ||
    "current" in value
  );
}

function formatFallbackScalar(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") {
    return new Intl.NumberFormat("en-US", {
      maximumFractionDigits: Math.abs(value) >= 100 ? 0 : 2,
    }).format(value);
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return value;

  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function titleize(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function escapeHtml(value: unknown) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getCompanyInfo(data: JsonRecord) {
  return asRecord(data.company_info || data.company);
}

function getAnthropicApiKey() {
  return process.env.ANTHROPIC_API_KEY || process.env.CLAUDE_API_KEY || "";
}

function logFinancialStage(input: FinancialStageLogInput) {
  const error = normalizeSafeError(input.error);

  console.info(
    JSON.stringify({
      stage: input.stage,
      fileCount: input.files?.length || 0,
      files: input.files?.map((file) => ({
        fileName: file.fileName,
        mimeType: file.fileType,
        fileSize: file.fileSize,
      })),
      generatedTextLength: input.generatedTextLength,
      claudeInputTextLength: input.claudeInputTextLength,
      claudeModel: input.claudeModel,
      error,
      ...input.extra,
    })
  );
}

function normalizeSafeError(error: unknown) {
  if (!error) return undefined;

  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
    };
  }

  if (isRecord(error)) {
    return {
      status: error.status,
      type: error.type,
      message: error.message,
      error: error.error,
    };
  }

  return { message: String(error) };
}

function getClaudeErrorCode(
  status: number,
  responseBody: unknown
): FinancialAnalysisErrorCode {
  const errorType = getClaudeResponseErrorType(responseBody);
  const message = getClaudeResponseErrorMessage(responseBody).toLowerCase();

  if (status === 401 || status === 403 || errorType === "authentication_error") {
    return "claude_auth_failure";
  }

  if (
    status === 404 ||
    errorType === "not_found_error" ||
    (message.includes("model") && message.includes("not"))
  ) {
    return "claude_model_not_found";
  }

  if (status === 429 || errorType === "rate_limit_error") {
    return "claude_rate_limit";
  }

  if (
    status === 413 ||
    message.includes("context") ||
    message.includes("too long") ||
    message.includes("maximum")
  ) {
    return "claude_context_too_large";
  }

  if (status === 400 || errorType === "invalid_request_error") {
    return "invalid_claude_request";
  }

  return "claude_analysis_failure";
}

function getClaudeErrorMessage(
  code: FinancialAnalysisErrorCode,
  responseBody: unknown
) {
  const apiMessage = getClaudeResponseErrorMessage(responseBody);

  if (apiMessage) return apiMessage;

  switch (code) {
    case "claude_auth_failure":
      return "Claude authentication failed";
    case "claude_model_not_found":
      return "Claude model was not found";
    case "claude_rate_limit":
      return "Claude rate limit reached";
    case "claude_context_too_large":
      return "Claude input is too large";
    case "claude_output_truncated":
      return "Claude response hit the output token limit before completing JSON";
    case "invalid_claude_request":
      return "Claude request is invalid";
    default:
      return "Claude analysis request failed";
  }
}

function getClaudeHttpStatus(code: FinancialAnalysisErrorCode, status: number) {
  switch (code) {
    case "claude_auth_failure":
      return 502;
    case "claude_model_not_found":
      return 502;
    case "claude_rate_limit":
      return 429;
    case "claude_context_too_large":
      return 413;
    case "claude_output_truncated":
      return 502;
    case "invalid_claude_request":
      return 400;
    default:
      return status >= 400 && status < 600 ? status : 502;
  }
}

function getClaudeResponseErrorType(responseBody: unknown) {
  if (!isRecord(responseBody)) return "";
  const directType = responseBody.type;
  const nestedError = responseBody.error;

  if (typeof directType === "string") return directType;
  if (isRecord(nestedError) && typeof nestedError.type === "string") {
    return nestedError.type;
  }

  return "";
}

function getClaudeResponseErrorMessage(responseBody: unknown) {
  if (typeof responseBody === "string") return responseBody;
  if (!isRecord(responseBody)) return "";

  if (typeof responseBody.message === "string") return responseBody.message;

  const nestedError = responseBody.error;

  if (isRecord(nestedError) && typeof nestedError.message === "string") {
    return nestedError.message;
  }

  return "";
}

function getDocumentLogInfo(document: FinancialStatementDocumentInput) {
  return {
    fileName: document.fileName,
    fileType: document.fileType,
    fileSize: typeof document.file.size === "number" ? document.file.size : undefined,
  };
}

function isPdfDocument(document: FinancialStatementDocumentInput) {
  return getFileExtension(document.fileName) === ".pdf" || isPdfMimeType(document.fileType);
}

function getGeneratedTextFileName(fileName: string) {
  const extension = getFileExtension(fileName);
  const withoutExtension = extension ? fileName.slice(0, -extension.length) : fileName;

  return `${withoutExtension || "financial-statement"}.txt`;
}

function slugifyFileName(fileName: string) {
  return fileName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

function parseUploadedAnalysisJson(text: string, fileName: string): JsonRecord {
  try {
    const parsed = JSON.parse(text);

    if (isRecord(parsed)) {
      return parsed;
    }

    throw new Error("JSON root is not an object");
  } catch (error) {
    throw new FinancialAnalysisError(
      "malformed_intermediate_output",
      `${fileName} is not valid financial analysis JSON`,
      400,
      error instanceof Error ? error.message : String(error)
    );
  }
}

async function readBlobAsText(blob: Blob, fileName: string) {
  try {
    return await blob.text();
  } catch (error) {
    throw new FinancialAnalysisError(
      "missing_financial_statement_input",
      `${fileName} could not be read as text`,
      400,
      error instanceof Error ? error.message : String(error)
    );
  }
}

async function readResponseBody(response: Response): Promise<unknown> {
  const text = await response.text();

  if (!text) return null;

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function extractClaudeText(responseBody: unknown) {
  if (!isRecord(responseBody) || !Array.isArray(responseBody.content)) {
    return "";
  }

  return responseBody.content
    .map((contentBlock) =>
      isRecord(contentBlock) && typeof contentBlock.text === "string"
        ? contentBlock.text
        : ""
    )
    .join("\n")
    .trim();
}

function getFileExtension(fileName: string) {
  const dotIndex = fileName.lastIndexOf(".");

  return dotIndex === -1 ? "" : fileName.slice(dotIndex).toLowerCase();
}

function isPdfMimeType(fileType: string | null) {
  return fileType === "application/pdf";
}

function isJsonMimeType(fileType: string | null) {
  return fileType === "application/json";
}

function isMarkdownMimeType(fileType: string | null) {
  return fileType === "text/markdown" || fileType === "text/x-markdown";
}

function isTextMimeType(fileType: string | null) {
  return fileType === "text/plain";
}

function getPositiveNumberEnv(name: string, fallback: number) {
  const value = Number(process.env[name]);

  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function hasAzureDocumentIntelligenceConfig() {
  return Boolean(
    process.env.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT &&
      process.env.AZURE_DOCUMENT_INTELLIGENCE_KEY
  );
}

function getErrorDetail(error: unknown): string {
  if (!(error instanceof Error)) return String(error);

  const cause = error.cause;
  if (cause instanceof Error && cause.message) {
    return `${error.message}: ${cause.message}`;
  }

  if (cause && typeof cause === "object" && "message" in cause) {
    const causeMessage = (cause as { message?: unknown }).message;
    if (typeof causeMessage === "string" && causeMessage) {
      return `${error.message}: ${causeMessage}`;
    }
  }

  return error.message;
}

function getStringFromRecord(value: unknown, key: string) {
  if (!isRecord(value)) return undefined;
  const item = value[key];

  return typeof item === "string" && item.trim() ? item : undefined;
}

function getNumberFromPath(value: unknown, path: string[]) {
  let current = value;

  for (const key of path) {
    if (!isRecord(current)) return null;
    current = current[key];
  }

  return typeof current === "number" ? current : null;
}

function getStringValue(value: unknown) {
  return typeof value === "string" ? value : "";
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function asRecord(value: unknown): JsonRecord {
  return isRecord(value) ? value : {};
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
