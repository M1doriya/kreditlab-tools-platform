import { createClient } from "@supabase/supabase-js";
import {
  convertFinancialPdfsToText,
  FinancialAnalysisError,
  estimateFinancialStatementAnalysisCostFromText,
  type FinancialStatementJsonInput,
  type FinancialStatementTextInput,
} from "@/lib/server/financial-statement-analysis";
import {
  CLAUDE_MODEL_PRICING,
  estimateClaudeCost,
  resolveClaudeModelId,
} from "@/lib/claude-models";
import { runToolIntegration, toolIntegrations } from "@/lib/server/tool-integrations";

const DOCUMENT_BUCKET = "case-documents";
const financialStatementTool = toolIntegrations.financialStatement;

export const runtime = "nodejs";
// Large audits on Opus 4.8 can take several minutes. Locally this is advisory;
// on a deployed host it's capped by the platform's function limit.
export const maxDuration = 800;

type GeneratedTextFileInput = {
  id?: unknown;
  fileName?: unknown;
  generatedFileName?: unknown;
  originalFileName?: unknown;
  text?: unknown;
  content?: unknown;
};

type CaseDocument = {
  id: string;
  file_name: string | null;
  file_path: string;
  file_type: string | null;
};

type AnalyzerResult = {
  success?: boolean;
  html?: string;
  report_html?: string;
  report?: unknown;
  reports?: unknown[];
  [key: string]: unknown;
};

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function getReportHtml(result: AnalyzerResult) {
  if (typeof result.html === "string") return result.html;
  if (typeof result.report_html === "string") return result.report_html;

  if (isRecord(result.report)) {
    if (typeof result.report.html === "string") return result.report.html;
    if (typeof result.report.report_html === "string") {
      return result.report.report_html;
    }
  }

  const firstReport = Array.isArray(result.reports) ? result.reports[0] : null;

  if (isRecord(firstReport) && typeof firstReport.html === "string") {
    return firstReport.html;
  }

  return null;
}

export function GET() {
  console.info(
    JSON.stringify({
      route: "/api/run-financial-analysis",
      method: "GET",
      stage: "renderer",
      envChecked: ["ANTHROPIC_MODEL", "ANTHROPIC_EFFORT"],
      defaultModel: resolveClaudeModelId(null, process.env.ANTHROPIC_MODEL),
    })
  );

  return Response.json({
    models: CLAUDE_MODEL_PRICING,
    defaultModel: resolveClaudeModelId(null, process.env.ANTHROPIC_MODEL),
  });
}

export async function POST(req: Request) {
  try {
    console.info(
      JSON.stringify({
        route: "/api/run-financial-analysis",
        method: req.method,
        stage: "claude_analysis",
      })
    );

    const body = (await req.json()) as {
      caseId?: unknown;
      documentIds?: unknown;
      generatedTextFiles?: unknown;
      model?: unknown;
      mode?: unknown;
    };
    const mode = body.mode === "estimate" ? "estimate" : "run";
    const caseId = typeof body.caseId === "string" ? body.caseId : "";
    const documentIds = Array.isArray(body.documentIds)
      ? body.documentIds.filter(
          (documentId): documentId is string => typeof documentId === "string"
        )
      : [];
    const generatedTextFiles = Array.isArray(body.generatedTextFiles)
      ? body.generatedTextFiles
      : [];
    const requestedModel = typeof body.model === "string" ? body.model : "";
    const selectedModel = resolveClaudeModelId(
      requestedModel,
      process.env.ANTHROPIC_MODEL
    );
    const uniqueDocumentIds = [...new Set(documentIds)];

    if (!selectedModel) {
      return Response.json(
        {
          error: "Selected Claude model is not allowed",
          code: "invalid_claude_model",
          detail: { model: requestedModel },
        },
        { status: 400 }
      );
    }

    if (!caseId) {
      return Response.json(
        { error: "caseId is required", code: "missing_input" },
        { status: 400 }
      );
    }

    if (uniqueDocumentIds.length === 0 && generatedTextFiles.length === 0) {
      return Response.json(
        {
          error: "Select at least one TXT, MD, or JSON input",
          code: "missing_input",
        },
        { status: 400 }
      );
    }

    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

    if (!supabaseUrl || !serviceRoleKey) {
      return Response.json(
        {
          error: "Missing server environment variables",
          code: "dashboard_execution_failure",
          hasSupabaseUrl: !!supabaseUrl,
          hasServiceRoleKey: !!serviceRoleKey,
        },
        { status: 500 }
      );
    }

    const supabaseAdmin = createClient(supabaseUrl, serviceRoleKey);
    const textDocuments: FinancialStatementTextInput[] = [];
    const jsonDocuments: FinancialStatementJsonInput[] = [];

    for (const generatedFile of generatedTextFiles) {
      const inputDocument = normalizeGeneratedTextFile(generatedFile);

      if ("json" in inputDocument) {
        jsonDocuments.push(inputDocument);
      } else {
        textDocuments.push(inputDocument);
      }
    }

    let orderedDocs: CaseDocument[] = [];

    if (uniqueDocumentIds.length > 0) {
      const { data: docsData, error: docsError } = await supabaseAdmin
        .from("case_documents")
        .select("*")
        .eq("case_id", caseId)
        .in("id", uniqueDocumentIds);

      if (docsError) {
        return Response.json(
          {
            error: docsError.message,
            code: "dashboard_execution_failure",
          },
          { status: 500 }
        );
      }

      const docs = (docsData || []) as CaseDocument[];
      const docsById = new Map(docs.map((doc) => [doc.id, doc]));
      const missingDocumentIds = uniqueDocumentIds.filter(
        (documentId) => !docsById.has(documentId)
      );

      if (missingDocumentIds.length > 0) {
        return Response.json(
          {
            error: "Some selected documents were not found for this case",
            code: "missing_input",
            missingDocumentIds,
          },
          { status: 404 }
        );
      }

      orderedDocs = uniqueDocumentIds.map(
        (documentId) => docsById.get(documentId) as CaseDocument
      );

      for (const doc of orderedDocs) {
        const extension = getFileExtension(doc.file_name || "");

        const { data: fileData, error: downloadError } =
          await supabaseAdmin.storage.from(DOCUMENT_BUCKET).download(doc.file_path);

        if (downloadError || !fileData) {
          return Response.json(
            {
              error:
                downloadError?.message ||
                `Failed to download ${doc.file_name || "document"}`,
              code: "missing_input",
            },
            { status: 500 }
          );
        }

        const fileName = doc.file_name || "financial-statement.txt";

        if (extension === ".pdf" || doc.file_type === "application/pdf") {
          if (mode === "estimate") {
            return Response.json(
              {
                error: "Convert the selected PDF with Azure OCR before requesting an exact estimate",
                code: "invalid_file_type",
                detail: { fileName: doc.file_name, fileType: doc.file_type },
              },
              { status: 400 }
            );
          }

          const conversion = await convertFinancialPdfsToText([
            {
              id: doc.id,
              fileName,
              fileType: doc.file_type || "application/pdf",
              file: fileData,
            },
          ]);

          for (const generatedFile of conversion.generatedTextFiles) {
            textDocuments.push({
              id: generatedFile.id,
              fileName: generatedFile.generatedFileName,
              fileType: generatedFile.fileType,
              text: generatedFile.text,
              sourceFileName: generatedFile.originalFileName,
              sourceKind: "generated_txt",
            });
          }

          continue;
        }

        const text = await fileData.text();

        if (
          extension === ".json" ||
          doc.file_type === "application/json"
        ) {
          jsonDocuments.push({
            id: doc.id,
            fileName,
            fileType: doc.file_type,
            json: parseAnalysisJsonDocument(text, fileName),
          });
          continue;
        }

        if (
          extension === ".txt" ||
          extension === ".md" ||
          doc.file_type === "text/plain" ||
          doc.file_type === "text/markdown" ||
          doc.file_type === "text/x-markdown"
        ) {
          textDocuments.push({
            id: doc.id,
            fileName,
            fileType: doc.file_type,
            text,
            sourceKind: extension === ".md" ? "markdown" : "text",
          });
          continue;
        }

        return Response.json(
          {
            error: "Only TXT, MD, or JSON files can be analyzed in Step 2",
            code: "invalid_file_type",
            detail: { fileName: doc.file_name, fileType: doc.file_type },
          },
          { status: 400 }
        );
      }
    }

    // Pre-flight estimate mode: ask analyze.py for the exact token count/cost via
    // --dry-run (the same engine the run uses), then return without spending money.
    if (mode === "estimate") {
      const estimate = await estimateFinancialStatementAnalysisCostFromText({
        textDocuments,
        jsonDocuments,
        model: selectedModel,
      });

      return Response.json({ success: true, estimate });
    }

    const claudeInputTextLength = textDocuments.reduce(
      (total, doc) => total + doc.text.length,
      0
    );
    const costEstimate = estimateClaudeCost({
      modelId: selectedModel,
      inputCharacters: claudeInputTextLength,
    });

    console.info(
      JSON.stringify({
        stage: "claude_analysis",
        fileCount: textDocuments.length + jsonDocuments.length,
        files: [
          ...textDocuments.map((doc) => ({
            fileName: doc.fileName,
            mimeType: doc.fileType,
            fileSize: doc.text.length,
          })),
          ...jsonDocuments.map((doc) => ({
            fileName: doc.fileName,
            mimeType: doc.fileType,
          })),
        ],
        hasAnthropicApiKey: Boolean(process.env.ANTHROPIC_API_KEY),
        hasClaudeApiKey: Boolean(process.env.CLAUDE_API_KEY),
        claudeModel: selectedModel,
        claudeInputTextLength,
        estimatedInputTokens: costEstimate.inputTokens,
        estimatedOutputTokens: costEstimate.outputTokens,
        estimatedCostUsd: costEstimate.totalCostUsd,
      })
    );

    let analyzerResult: AnalyzerResult;

    try {
      analyzerResult = (await runToolIntegration("financialStatement", {
        textDocuments,
        jsonDocuments,
        model: selectedModel,
      })) as AnalyzerResult;
    } catch (error) {
      if (error instanceof FinancialAnalysisError) {
        console.error(
          JSON.stringify({
            stage: "claude_analysis",
            code: error.code,
            message: error.message,
            detail: error.detail,
            stack: error.stack,
          })
        );

        return Response.json(
          {
            error: error.message,
            code: error.code,
            detail: error.detail,
          },
          { status: error.status }
        );
      }

      throw error;
    }

    const reportHtml = getReportHtml(analyzerResult);

    if (!reportHtml || typeof reportHtml !== "string") {
      return Response.json(
        {
          error: "Analyzer did not return report HTML",
          code: "claude_analysis_failure",
        },
        { status: 502 }
      );
    }

    const normalizedAnalyzerResult = {
      ...analyzerResult,
      html: reportHtml,
    };

    const { data: savedReport, error: saveError } = await supabaseAdmin
      .from("case_analysis_reports")
      .insert([
        {
          case_id: caseId,
          analysis_type: financialStatementTool.analysisType,
          report_html: reportHtml,
          report_json: {
            ...normalizedAnalyzerResult,
            source_document_ids: uniqueDocumentIds,
            generated_text_files: textDocuments
              .filter((doc) => doc.sourceKind === "generated_txt")
              .map((doc) => ({
                id: doc.id,
                file_name: doc.fileName,
                source_file_name: doc.sourceFileName,
                text_length: doc.text.length,
              })),
            source_documents: orderedDocs.map((doc) => ({
              id: doc.id,
              file_name: doc.file_name,
              file_type: doc.file_type,
            })),
            tool_name: financialStatementTool.displayName,
            selected_claude_model: selectedModel,
          },
        },
      ])
      .select()
      .single();

    if (saveError) {
      return Response.json(
        {
          error: saveError.message,
          code: "dashboard_execution_failure",
        },
        { status: 500 }
      );
    }

    return Response.json({
      success: true,
      message: "Analysis completed",
      report: normalizedAnalyzerResult,
      savedReport,
    });
  } catch (error) {
    if (error instanceof FinancialAnalysisError) {
      console.error(
        JSON.stringify({
          stage: "claude_analysis",
          code: error.code,
          message: error.message,
          detail: error.detail,
          stack: error.stack,
        })
      );

      return Response.json(
        {
          error: error.message,
          code: error.code,
          detail: error.detail,
        },
        { status: error.status }
      );
    }

    console.error(
      JSON.stringify({
        stage: "claude_analysis",
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
      })
    );

    return Response.json(
      {
        error: "Failed to run analysis",
        code: "dashboard_execution_failure",
        detail: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

function normalizeGeneratedTextFile(
  generatedFile: unknown
): FinancialStatementTextInput | FinancialStatementJsonInput {
  if (!isRecord(generatedFile)) {
    throw new FinancialAnalysisError(
      "missing_input",
      "Generated TXT input is malformed",
      400
    );
  }

  const fileName =
    getString(generatedFile.generatedFileName) ||
    getString(generatedFile.fileName) ||
    "generated-financial-statement.txt";
  const text = getString(generatedFile.text) || getString(generatedFile.content);

  if (!text.trim()) {
    throw new FinancialAnalysisError(
      "text_file_empty",
      `${fileName} is empty`,
      400
    );
  }

  if (getFileExtension(fileName) === ".json") {
    return {
      id: getString(generatedFile.id),
      fileName,
      fileType: "application/json",
      json: parseAnalysisJsonDocument(text, fileName),
    };
  }

  return {
    id: getString(generatedFile.id),
    fileName,
    fileType: "text/plain",
    text,
    sourceFileName: getString(generatedFile.originalFileName),
    sourceKind: "generated_txt",
  };
}

function getFileExtension(fileName: string) {
  const dotIndex = fileName.lastIndexOf(".");

  return dotIndex === -1 ? "" : fileName.slice(dotIndex).toLowerCase();
}

function getString(value: unknown) {
  return typeof value === "string" ? value : "";
}

function parseAnalysisJsonDocument(text: string, fileName: string): JsonRecord {
  try {
    const parsed = JSON.parse(text);

    if (!isRecord(parsed)) {
      throw new Error("JSON root must be an object");
    }

    return parsed;
  } catch (error) {
    throw new FinancialAnalysisError(
      "malformed_intermediate_output",
      `${fileName} is not valid financial analysis JSON`,
      400,
      error instanceof Error ? error.message : String(error)
    );
  }
}
