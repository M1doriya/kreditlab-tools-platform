import { createClient } from "@supabase/supabase-js";
import { bankStatementAnalysisSection } from "@/lib/bank-statement-analysis-config";
import {
  BankAnalysisError,
  runBankStatementAnalysisFromPdfs,
  type BankStatementDocumentInput,
} from "@/lib/server/bank-statement-analysis";

const DOCUMENT_BUCKET = "case-documents";
const bankStatementTool = bankStatementAnalysisSection.tool;

export const runtime = "nodejs";
export const maxDuration = 800;

type CaseDocument = {
  id: string;
  file_name: string | null;
  file_path: string;
  file_type: string | null;
};

type AnalyzerResult = {
  html?: unknown;
  report_html?: unknown;
  report?: unknown;
  [key: string]: unknown;
};

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const caseId = getString(formData.get("caseId"));
    const bankName = getString(formData.get("bankName"));
    const pdfPassword = getString(formData.get("pdfPassword"));
    const companyNameOverride = getString(formData.get("companyNameOverride"));
    const documentIds = parseDocumentIds(formData);
    const uploadedFiles = formData
      .getAll("files")
      .filter((file): file is File => file instanceof File);

    if (!caseId) {
      return Response.json(
        { error: "caseId is required", code: "missing_input" },
        { status: 400 }
      );
    }

    const documents: BankStatementDocumentInput[] = [];
    let selectedDocs: CaseDocument[] = [];

    for (const file of uploadedFiles) {
      if (!isPdfFile(file.name, file.type)) {
        return Response.json(
          {
            error: "Bank Statement Analyzer only accepts PDF files",
            code: "invalid_file_type",
            detail: { fileName: file.name, fileType: file.type },
          },
          { status: 400 }
        );
      }

      documents.push({
        fileName: file.name,
        fileType: file.type || "application/pdf",
        file,
      });
    }

    if (documentIds.length > 0) {
      const supabaseAdmin = getSupabaseAdminClient();
      const { data: docsData, error: docsError } = await supabaseAdmin
        .from("case_documents")
        .select("*")
        .eq("case_id", caseId)
        .in("id", documentIds);

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
      const missingDocumentIds = documentIds.filter(
        (documentId) => !docsById.has(documentId)
      );

      if (missingDocumentIds.length > 0) {
        return Response.json(
          {
            error: "Some selected PDFs were not found for this case",
            code: "missing_input",
            missingDocumentIds,
          },
          { status: 404 }
        );
      }

      selectedDocs = documentIds.map(
        (documentId) => docsById.get(documentId) as CaseDocument
      );

      for (const doc of selectedDocs) {
        if (!isPdfFile(doc.file_name || "", doc.file_type || "")) {
          return Response.json(
            {
              error: "Bank Statement Analyzer only accepts PDF files",
              code: "invalid_file_type",
              detail: { fileName: doc.file_name, fileType: doc.file_type },
            },
            { status: 400 }
          );
        }

        const { data: fileData, error: downloadError } =
          await supabaseAdmin.storage.from(DOCUMENT_BUCKET).download(doc.file_path);

        if (downloadError || !fileData) {
          return Response.json(
            {
              error:
                downloadError?.message ||
                `Failed to download ${doc.file_name || "PDF"}`,
              code: "missing_input",
            },
            { status: 500 }
          );
        }

        documents.push({
          id: doc.id,
          fileName: doc.file_name || "bank-statement.pdf",
          fileType: doc.file_type || "application/pdf",
          file: fileData,
        });
      }
    }

    if (documents.length === 0) {
      return Response.json(
        {
          error: "Upload or select at least one bank statement PDF",
          code: "missing_input",
        },
        { status: 400 }
      );
    }

    console.info(
      JSON.stringify({
        route: "/api/run-bank-analysis",
        bankName,
        fileCount: documents.length,
        files: documents.map((document) => ({
          fileName: document.fileName,
          mimeType: document.fileType,
          fileSize:
            typeof document.file.size === "number" ? document.file.size : undefined,
        })),
      })
    );

    const analyzerResult = (await runBankStatementAnalysisFromPdfs({
      bankName,
      documents,
      pdfPassword,
      companyNameOverride,
    })) as AnalyzerResult;
    const reportHtml = getReportHtml(analyzerResult);

    if (!reportHtml) {
      return Response.json(
        {
          error: "Analyzer did not return report HTML",
          code: "bank_statement_analysis_failure",
        },
        { status: 502 }
      );
    }

    const normalizedAnalyzerResult = {
      ...analyzerResult,
      html: reportHtml,
    };
    const supabaseAdmin = getSupabaseAdminClient();
    const { data: savedReport, error: saveError } = await supabaseAdmin
      .from("case_analysis_reports")
      .insert([
        {
          case_id: caseId,
          analysis_type: bankStatementTool.analysisType,
          report_html: reportHtml,
          report_json: {
            ...normalizedAnalyzerResult,
            bank_name: bankName,
            source_document_ids: documentIds,
            source_documents: selectedDocs.map((doc) => ({
              id: doc.id,
              file_name: doc.file_name,
              file_type: doc.file_type,
            })),
            uploaded_files: uploadedFiles.map((file) => ({
              file_name: file.name,
              file_type: file.type,
              file_size: file.size,
            })),
            tool_name: bankStatementTool.displayName,
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
      message: "Bank statement analysis completed",
      report: normalizedAnalyzerResult,
      savedReport,
    });
  } catch (error) {
    if (error instanceof BankAnalysisError) {
      console.error(
        JSON.stringify({
          route: "/api/run-bank-analysis",
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
        route: "/api/run-bank-analysis",
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
      })
    );

    return Response.json(
      {
        error: "Failed to run bank statement analysis",
        code: "dashboard_execution_failure",
        detail: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

function getSupabaseAdminClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !serviceRoleKey) {
    throw new BankAnalysisError(
      "dashboard_execution_failure",
      "Missing server environment variables",
      500,
      {
        hasSupabaseUrl: !!supabaseUrl,
        hasServiceRoleKey: !!serviceRoleKey,
      }
    );
  }

  return createClient(supabaseUrl, serviceRoleKey);
}

function parseDocumentIds(formData: FormData) {
  const directIds = formData
    .getAll("documentIds")
    .filter((value): value is string => typeof value === "string");
  const jsonValue = getString(formData.get("documentIdsJson"));

  if (!jsonValue) return [...new Set(directIds)];

  try {
    const parsed = JSON.parse(jsonValue);
    const jsonIds = Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];

    return [...new Set([...directIds, ...jsonIds])];
  } catch {
    return [...new Set(directIds)];
  }
}

function getReportHtml(result: AnalyzerResult) {
  if (typeof result.html === "string") return result.html;
  if (typeof result.report_html === "string") return result.report_html;

  const nestedReport = result.report;
  if (isRecord(nestedReport) && typeof nestedReport.html === "string") {
    return nestedReport.html;
  }

  return "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPdfFile(fileName: string, fileType: string | null) {
  return fileName.toLowerCase().endsWith(".pdf") || fileType === "application/pdf";
}

function getString(value: FormDataEntryValue | null) {
  return typeof value === "string" ? value : "";
}
