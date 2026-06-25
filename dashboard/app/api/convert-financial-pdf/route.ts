import { createClient } from "@supabase/supabase-js";
import {
  convertFinancialPdfsToText,
  FinancialAnalysisError,
  type FinancialStatementDocumentInput,
} from "@/lib/server/financial-statement-analysis";

const DOCUMENT_BUCKET = "case-documents";

export const runtime = "nodejs";
export const maxDuration = 300;

type CaseDocument = {
  id: string;
  file_name: string | null;
  file_path: string;
  file_type: string | null;
};

export async function POST(req: Request) {
  try {
    console.info(
      JSON.stringify({
        route: "/api/convert-financial-pdf",
        method: req.method,
        stage: "ocr_conversion",
      })
    );

    const formData = await req.formData();
    const caseId = getString(formData.get("caseId"));
    const uploadedFiles = formData
      .getAll("files")
      .filter((file): file is File => file instanceof File);
    const documentIds = parseDocumentIds(formData);
    const documents: FinancialStatementDocumentInput[] = [];
    let selectedDocs: CaseDocument[] = [];

    for (const file of uploadedFiles) {
      if (!isPdfFile(file.name, file.type)) {
        return Response.json(
          {
            error: "Step 1 only accepts PDF files",
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
      if (!caseId) {
        return Response.json(
          {
            error: "caseId is required when converting saved case documents",
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

      selectedDocs = documentIds.map((documentId) => docsById.get(documentId) as CaseDocument);

      for (const doc of selectedDocs) {
        if (!isPdfFile(doc.file_name || "", doc.file_type || "")) {
          return Response.json(
            {
              error: "Step 1 only accepts PDF files",
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
              code: "pdf_upload_failure",
            },
            { status: 500 }
          );
        }

        documents.push({
          id: doc.id,
          fileName: doc.file_name || "financial-statement.pdf",
          fileType: doc.file_type || "application/pdf",
          file: fileData,
        });
      }
    }

    if (documents.length === 0) {
      return Response.json(
        {
          error: "Upload or select at least one PDF file",
          code: "missing_input",
        },
        { status: 400 }
      );
    }

    console.info(
      JSON.stringify({
        stage: "ocr_conversion",
        fileCount: documents.length,
        files: documents.map((document) => ({
          fileName: document.fileName,
          mimeType: document.fileType,
          fileSize:
            typeof document.file.size === "number" ? document.file.size : undefined,
        })),
        hasServiceApiKey: Boolean(process.env.SERVICE_API_KEY),
        hasAzureDocumentIntelligenceEndpoint: Boolean(
          process.env.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
        ),
        hasAzureDocumentIntelligenceKey: Boolean(
          process.env.AZURE_DOCUMENT_INTELLIGENCE_KEY
        ),
      })
    );

    const result = await convertFinancialPdfsToText(documents);

    return Response.json(result);
  } catch (error) {
    if (error instanceof FinancialAnalysisError) {
      console.error(
        JSON.stringify({
          stage: "ocr_conversion",
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
        stage: "ocr_conversion",
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
      })
    );

    return Response.json(
      {
        error: "Failed to convert PDF with Azure OCR",
        code: "ocr_extraction_failure",
        detail: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
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

function isPdfFile(fileName: string, fileType: string | null) {
  return fileName.toLowerCase().endsWith(".pdf") || fileType === "application/pdf";
}

function getString(value: FormDataEntryValue | null) {
  return typeof value === "string" ? value : "";
}
