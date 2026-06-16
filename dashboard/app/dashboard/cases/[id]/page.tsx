"use client";

import { useEffect, useState } from "react";
import { notFound, useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import { bankStatementAnalysisSection } from "@/lib/bank-statement-analysis-config";
import { financialStatementAnalysisSection } from "@/lib/financial-statement-analysis-config";
import {
  Building2,
  CalendarDays,
  CircleDot,
  ClipboardList,
  Download,
  FileText,
  Loader2,
  Pencil,
  User,
  Trash2,
} from "lucide-react";
import {
  FinancialStatementAnalysisErrorBoundary,
  FinancialStatementAnalysisSection,
} from "./financial-statement-analysis-section";
import {
  BankStatementAnalysisErrorBoundary,
  BankStatementAnalysisSection,
} from "./bank-statement-analysis-section";

type CaseItem = {
  id: string;
  case_code: string | null;
  company_name: string;
  client_name: string;
  email: string;
  phone: string | null;
  industry: string;
  requested_amount: number;
  loan_purpose: string | null;
  status: string | null;
  assigned_to: string | null;
  updated_at: string | null;
  ssm_registration_id: string | null;
  initial_notes: string | null;
  annual_revenue: number | null;
  employee_count: number | null;
  created_at?: string | null;
};

type CaseDocument = {
  id: string;
  case_id: string;
  file_name: string;
  file_path: string;
  file_type: string | null;
  document_type: string | null;
  uploaded_at: string | null;
};

type TabKey =
  | "overview"
  | "documents"
  | "analysis"
  | "notes"
  | "pipeline"
  | "report";

type AnalysisReport = {
  report_html: string;
  report_json: unknown;
};

type FinancialExportFormat = "html" | "pdf" | "excel";

export default function CaseDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [caseData, setCaseData] = useState<CaseItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabKey>("overview");

  const [documents, setDocuments] = useState<CaseDocument[]>([]);
  const [financialAnalysisReport, setFinancialAnalysisReport] =
    useState<AnalysisReport | null>(null);
  const [bankAnalysisReport, setBankAnalysisReport] =
    useState<AnalysisReport | null>(null);

  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState("");
  const [exportingReportFormat, setExportingReportFormat] =
    useState<FinancialExportFormat | null>(null);
  const [reportExportError, setReportExportError] = useState("");

  const fetchDocuments = async () => {
    if (!id) return;

    const { data, error } = await supabase
      .from("case_documents")
      .select("*")
      .eq("case_id", id)
      .order("uploaded_at", { ascending: false });

    if (!error && data) {
      setDocuments(data);
    }
  };

  const fetchLatestAnalysisReport = async (
    analysisType: string,
    onLoaded: (report: AnalysisReport | null) => void
  ) => {
    if (!id) return;

    const { data, error } = await supabase
      .from("case_analysis_reports")
      .select("*")
      .eq("case_id", id)
      .eq("analysis_type", analysisType)
      .order("created_at", { ascending: false })
      .limit(1)
      .single();

    if (!error && data) {
      onLoaded(data);
    } else {
      onLoaded(null);
    }
  };

  const fetchAnalysisReports = async () => {
    await Promise.all([
      fetchLatestAnalysisReport(
        financialStatementAnalysisSection.tool.analysisType,
        setFinancialAnalysisReport
      ),
      fetchLatestAnalysisReport(
        bankStatementAnalysisSection.tool.analysisType,
        setBankAnalysisReport
      ),
    ]);
  };

  useEffect(() => {
    const fetchCase = async () => {
      const { data, error } = await supabase
        .from("cases")
        .select("*")
        .eq("id", id)
        .single();

      if (error || !data) {
        setLoading(false);
        return;
      }

      setCaseData(data);
      setLoading(false);
    };

    void fetchCase();
    const timer = window.setTimeout(() => {
      void fetchDocuments();
      void fetchAnalysisReports();
    }, 0);

    return () => window.clearTimeout(timer);
  }, [id]);

  if (!loading && !caseData) {
    notFound();
  }

  const handleUpdateCase = async () => {
    if (!caseData) return;

    setSaving(true);

    const { error } = await supabase
      .from("cases")
      .update({
        company_name: caseData.company_name,
        client_name: caseData.client_name,
        email: caseData.email,
        phone: caseData.phone,
        industry: caseData.industry,
        requested_amount: Number(caseData.requested_amount || 0),
        loan_purpose: caseData.loan_purpose,
        status: caseData.status,
        assigned_to: caseData.assigned_to,
        ssm_registration_id: caseData.ssm_registration_id,
        initial_notes: caseData.initial_notes,
        annual_revenue: caseData.annual_revenue
          ? Number(caseData.annual_revenue)
          : null,
        employee_count: caseData.employee_count
          ? Number(caseData.employee_count)
          : null,
        updated_at: new Date().toISOString(),
      })
      .eq("id", caseData.id);

    setSaving(false);

    if (error) {
      alert(error.message);
      return;
    }

    setIsEditing(false);
    alert("Case updated successfully.");
  };

  const handleDeleteCase = async () => {
    if (!caseData) return;

    const confirmDelete = window.confirm(
      `Are you sure you want to delete ${caseData.company_name}?`
    );

    if (!confirmDelete) return;

    setDeleting(true);

    const { error } = await supabase
      .from("cases")
      .delete()
      .eq("id", caseData.id);

    setDeleting(false);

    if (error) {
      alert(error.message);
      return;
    }

    router.push("/dashboard/cases");
    router.refresh();
  };

  const handleDeleteDocument = async (document: CaseDocument) => {
    const confirmDelete = window.confirm(
      `Remove ${document.file_name} from this case?`
    );

    if (!confirmDelete) return;

    setDeletingDocumentId(document.id);

    try {
      const response = await fetch("/api/delete-case-document", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          caseId: id,
          documentId: document.id,
        }),
      });
      const result = (await response.json().catch(() => ({}))) as {
        error?: unknown;
        code?: unknown;
      };

      if (!response.ok) {
        throw new Error(getApiErrorMessage(result, "Document could not be removed"));
      }

      setDocuments((current) =>
        current.filter((item) => item.id !== document.id)
      );
    } catch (error) {
      alert(error instanceof Error ? error.message : String(error));
    } finally {
      setDeletingDocumentId("");
    }
  };

  const downloadAnalysisReport = async (format: FinancialExportFormat) => {
    if (!financialAnalysisReport) return;

    setExportingReportFormat(format);
    setReportExportError("");

    try {
      const response = await fetch("/api/export-financial-analysis", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          format,
          report: financialAnalysisReport.report_json || financialAnalysisReport,
          fileName: caseData?.company_name || "financial-analysis",
        }),
      });

      if (!response.ok) {
        const result = (await response.json().catch(() => ({}))) as {
          error?: unknown;
          code?: unknown;
        };
        throw new Error(
          getApiErrorMessage(result, "Financial analysis export failed")
        );
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
      setReportExportError(error instanceof Error ? error.message : String(error));
    } finally {
      setExportingReportFormat(null);
    }
  };

  const getStatusStyles = (status: string | null) => {
    switch (status) {
      case "Approved":
        return "bg-green-100 text-green-700 border-green-200";
      case "Rejected":
        return "bg-red-100 text-red-700 border-red-200";
      case "In Progress":
        return "bg-yellow-100 text-yellow-700 border-yellow-200";
      case "Under Review":
        return "bg-purple-100 text-purple-700 border-purple-200";
      default:
        return "bg-cyan-100 text-cyan-700 border-cyan-200";
    }
  };

  const formatCurrency = (amount: number | null) => {
    return new Intl.NumberFormat("en-MY", {
      style: "currency",
      currency: "MYR",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount || 0);
  };

  const formatDate = (value: string | null | undefined) => {
    if (!value) return "-";
    return new Date(value).toLocaleDateString();
  };

  const renderTabButton = (tab: TabKey, label: string) => {
    const active = activeTab === tab;

    return (
      <button
        type="button"
        onClick={() => setActiveTab(tab)}
        className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
          active
            ? "bg-white text-slate-900 shadow-sm"
            : "text-slate-600 hover:text-slate-900"
        }`}
      >
        {label}
      </button>
    );
  };

  const inputClass =
    "mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none focus:border-cyan-400";

  return (
    <main className="min-h-screen bg-slate-100 p-6">
      <div className="mx-auto max-w-6xl">
        {loading ? (
          <div className="rounded-2xl bg-white p-6 shadow-sm">Loading...</div>
        ) : (
          <>
            <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <Link
                href="/dashboard/cases"
                className="inline-flex items-center gap-2 text-sm text-slate-600 hover:text-slate-900"
              >
                â† Back to Cases
              </Link>

              <div className="flex flex-wrap items-center gap-3">
                <span
                  className={`inline-flex rounded-full border px-3 py-1 text-sm font-medium ${getStatusStyles(
                    caseData?.status ?? null
                  )}`}
                >
                  {caseData?.status || "New"}
                </span>

                <span className="text-3xl font-bold text-slate-900">
                  {formatCurrency(caseData?.requested_amount || 0)}
                </span>

                <button
                  type="button"
                  onClick={() => setIsEditing((prev) => !prev)}
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  <Pencil className="h-4 w-4" />
                  {isEditing ? "Cancel" : "Edit"}
                </button>

                <button
                  type="button"
                  onClick={handleDeleteCase}
                  disabled={deleting}
                  className="inline-flex items-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-100 disabled:opacity-60"
                >
                  <Trash2 className="h-4 w-4" />
                  {deleting ? "Deleting..." : "Delete"}
                </button>
              </div>
            </div>

            <div className="mb-4">
              {isEditing && caseData ? (
                <input
                  value={caseData.company_name}
                  onChange={(e) =>
                    setCaseData({ ...caseData, company_name: e.target.value })
                  }
                  className="w-full rounded-xl border border-slate-300 px-4 py-3 text-3xl font-bold text-slate-900 outline-none focus:border-cyan-400"
                />
              ) : (
                <h1 className="text-3xl font-bold text-slate-900">
                  {caseData?.company_name}
                </h1>
              )}

              <p className="mt-1 text-sm text-slate-500">
                {caseData?.case_code || caseData?.id} â€¢ {caseData?.client_name}
              </p>
            </div>

            <div className="mb-6 flex flex-wrap gap-2 rounded-2xl bg-slate-200 p-2">
              {renderTabButton("overview", "Overview")}
              {renderTabButton("documents", "Documents")}
              {renderTabButton("analysis", "Analysis")}
              {renderTabButton("notes", "Notes")}
              {renderTabButton("pipeline", "Pipeline")}
              {renderTabButton("report", "Report")}
            </div>

            {isEditing && (
              <div className="mb-6 flex justify-end">
                <button
                  type="button"
                  onClick={handleUpdateCase}
                  disabled={saving}
                  className="rounded-xl bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-900 hover:bg-cyan-300 disabled:opacity-60"
                >
                  {saving ? "Saving..." : "Save Changes"}
                </button>
              </div>
            )}

            {activeTab === "overview" && caseData && (
              <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
                <div className="space-y-6">
                  <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                    <div className="mb-6 flex items-center gap-2">
                      <Building2 className="h-5 w-5 text-cyan-500" />
                      <h2 className="text-xl font-semibold text-slate-900">
                        Company Information
                      </h2>
                    </div>

                    <div className="grid gap-6 md:grid-cols-2">
                      <div>
                        <p className="text-sm text-slate-500">Company Name</p>
                        {isEditing ? (
                          <input
                            value={caseData.company_name}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                company_name: e.target.value,
                              })
                            }
                            className={inputClass}
                          />
                        ) : (
                          <p className="mt-1 text-xl font-semibold text-slate-900">
                            {caseData.company_name}
                          </p>
                        )}
                      </div>

                      <div>
                        <p className="text-sm text-slate-500">Employee Count</p>
                        {isEditing ? (
                          <input
                            type="number"
                            value={caseData.employee_count ?? ""}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                employee_count: e.target.value
                                  ? Number(e.target.value)
                                  : null,
                              })
                            }
                            className={inputClass}
                          />
                        ) : (
                          <p className="mt-1 text-xl font-semibold text-slate-900">
                            {caseData.employee_count ?? "-"}
                          </p>
                        )}
                      </div>

                      <div>
                        <p className="text-sm text-slate-500">Industry</p>
                        {isEditing ? (
                          <input
                            value={caseData.industry}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                industry: e.target.value,
                              })
                            }
                            className={inputClass}
                          />
                        ) : (
                          <p className="mt-1 text-xl font-semibold text-slate-900">
                            {caseData.industry}
                          </p>
                        )}
                      </div>

                      <div>
                        <p className="text-sm text-slate-500">Loan Purpose</p>
                        {isEditing ? (
                          <input
                            value={caseData.loan_purpose || ""}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                loan_purpose: e.target.value,
                              })
                            }
                            className={inputClass}
                          />
                        ) : (
                          <p className="mt-1 text-xl font-semibold text-slate-900">
                            {caseData.loan_purpose || "-"}
                          </p>
                        )}
                      </div>

                      <div>
                        <p className="text-sm text-slate-500">Annual Revenue</p>
                        {isEditing ? (
                          <input
                            type="number"
                            value={caseData.annual_revenue ?? ""}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                annual_revenue: e.target.value
                                  ? Number(e.target.value)
                                  : null,
                              })
                            }
                            className={inputClass}
                          />
                        ) : (
                          <p className="mt-1 text-xl font-semibold text-slate-900">
                            {caseData.annual_revenue
                              ? formatCurrency(caseData.annual_revenue)
                              : "-"}
                          </p>
                        )}
                      </div>

                      <div>
                        <p className="text-sm text-slate-500">Loan Amount</p>
                        {isEditing ? (
                          <input
                            type="number"
                            value={caseData.requested_amount}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                requested_amount: Number(e.target.value),
                              })
                            }
                            className={inputClass}
                          />
                        ) : (
                          <p className="mt-1 text-3xl font-bold text-slate-900">
                            {formatCurrency(caseData.requested_amount)}
                          </p>
                        )}
                      </div>
                    </div>
                  </section>

                  <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                    <div className="mb-6 flex items-center gap-2">
                      <User className="h-5 w-5 text-cyan-500" />
                      <h2 className="text-xl font-semibold text-slate-900">
                        Contact Information
                      </h2>
                    </div>

                    <div className="grid gap-4 md:grid-cols-3">
                      <div className="rounded-xl bg-slate-50 p-4">
                        <p className="text-sm text-slate-500">Client Name</p>
                        {isEditing ? (
                          <input
                            value={caseData.client_name}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                client_name: e.target.value,
                              })
                            }
                            className={inputClass}
                          />
                        ) : (
                          <p className="text-lg font-semibold text-slate-900">
                            {caseData.client_name}
                          </p>
                        )}
                      </div>

                      <div className="rounded-xl bg-slate-50 p-4">
                        <p className="text-sm text-slate-500">Email</p>
                        {isEditing ? (
                          <input
                            type="email"
                            value={caseData.email}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                email: e.target.value,
                              })
                            }
                            className={inputClass}
                          />
                        ) : (
                          <p className="break-all text-lg font-semibold text-slate-900">
                            {caseData.email}
                          </p>
                        )}
                      </div>

                      <div className="rounded-xl bg-slate-50 p-4">
                        <p className="text-sm text-slate-500">Phone</p>
                        {isEditing ? (
                          <input
                            value={caseData.phone || ""}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                phone: e.target.value,
                              })
                            }
                            className={inputClass}
                          />
                        ) : (
                          <p className="text-lg font-semibold text-slate-900">
                            {caseData.phone || "-"}
                          </p>
                        )}
                      </div>
                    </div>
                  </section>

                  <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                    <div className="mb-6 flex items-center gap-2">
                      <ClipboardList className="h-5 w-5 text-cyan-500" />
                      <h2 className="text-xl font-semibold text-slate-900">
                        Registration Details
                      </h2>
                    </div>

                    <div className="grid gap-6 md:grid-cols-2">
                      <div>
                        <p className="text-sm text-slate-500">
                          SSM Registration ID
                        </p>
                        {isEditing ? (
                          <input
                            value={caseData.ssm_registration_id || ""}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                ssm_registration_id: e.target.value,
                              })
                            }
                            className={inputClass}
                          />
                        ) : (
                          <p className="mt-1 text-lg font-semibold text-slate-900">
                            {caseData.ssm_registration_id || "-"}
                          </p>
                        )}
                      </div>

                      <div>
                        <p className="text-sm text-slate-500">Case Code</p>
                        <p className="mt-1 text-lg font-semibold text-slate-900">
                          {caseData.case_code || caseData.id}
                        </p>
                      </div>
                    </div>
                  </section>
                </div>

                <div className="space-y-6">
                  <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                    <h2 className="mb-5 text-xl font-semibold text-slate-900">
                      Case Summary
                    </h2>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between gap-4">
                        <p className="text-slate-500">Assigned To</p>
                        {isEditing ? (
                          <input
                            value={caseData.assigned_to || ""}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                assigned_to: e.target.value,
                              })
                            }
                            className="w-40 rounded-xl border border-slate-300 px-3 py-2 text-sm"
                          />
                        ) : (
                          <p className="font-medium text-slate-900">
                            {caseData.assigned_to || "-"}
                          </p>
                        )}
                      </div>

                      <div className="flex items-center justify-between gap-4">
                        <p className="text-slate-500">Status</p>
                        {isEditing ? (
                          <select
                            value={caseData.status || "New"}
                            onChange={(e) =>
                              setCaseData({
                                ...caseData,
                                status: e.target.value,
                              })
                            }
                            className="rounded-xl border border-slate-300 px-3 py-2 text-sm"
                          >
                            <option value="New">New</option>
                            <option value="In Progress">In Progress</option>
                            <option value="Under Review">Under Review</option>
                            <option value="Approved">Approved</option>
                            <option value="Rejected">Rejected</option>
                          </select>
                        ) : (
                          <p className="font-medium text-slate-900">
                            {caseData.status || "New"}
                          </p>
                        )}
                      </div>
                    </div>
                  </section>

                  <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                    <div className="mb-5 flex items-center gap-2">
                      <CalendarDays className="h-5 w-5 text-cyan-500" />
                      <h2 className="text-xl font-semibold text-slate-900">
                        Timeline
                      </h2>
                    </div>

                    <div className="space-y-6">
                      <div className="flex gap-3">
                        <CircleDot className="mt-1 h-4 w-4 text-cyan-500" />
                        <div>
                          <p className="font-semibold text-slate-900">
                            Created
                          </p>
                          <p className="text-sm text-slate-500">
                            {formatDate(
                              caseData.created_at || caseData.updated_at
                            )}
                          </p>
                        </div>
                      </div>

                      <div className="flex gap-3">
                        <CircleDot className="mt-1 h-4 w-4 text-cyan-500" />
                        <div>
                          <p className="font-semibold text-slate-900">
                            Last Updated
                          </p>
                          <p className="text-sm text-slate-500">
                            {formatDate(caseData.updated_at)}
                          </p>
                        </div>
                      </div>
                    </div>
                  </section>
                </div>
              </div>
            )}

            {activeTab === "documents" && caseData && (
              <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-slate-900">
                      Case Documents
                    </h2>
                    <p className="mt-1 text-sm text-slate-600">
                      Files attached to this case and available for analysis.
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={() => void fetchDocuments()}
                    className="inline-flex items-center justify-center rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                  >
                    Refresh
                  </button>
                </div>

                {documents.length === 0 ? (
                  <div className="mt-6 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center">
                    <FileText className="mx-auto h-8 w-8 text-slate-400" />
                    <p className="mt-3 font-medium text-slate-700">
                      No documents attached yet
                    </p>
                  </div>
                ) : (
                  <div className="mt-6 overflow-hidden rounded-xl border border-slate-200">
                    <table className="min-w-full text-sm">
                      <thead className="bg-slate-50 text-left text-slate-600">
                        <tr>
                          <th className="px-4 py-3 font-medium">File</th>
                          <th className="px-4 py-3 font-medium">Type</th>
                          <th className="px-4 py-3 font-medium">Uploaded</th>
                          <th className="px-4 py-3 text-right font-medium">
                            Action
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {documents.map((document) => {
                          const deletingThisDocument =
                            deletingDocumentId === document.id;

                          return (
                            <tr
                              key={document.id}
                              className="border-t border-slate-200"
                            >
                              <td className="px-4 py-3">
                                <div className="font-medium text-slate-900">
                                  {document.file_name}
                                </div>
                                <div className="mt-1 max-w-md truncate text-xs text-slate-500">
                                  {document.file_path}
                                </div>
                              </td>
                              <td className="px-4 py-3 text-slate-600">
                                {document.document_type ||
                                  document.file_type ||
                                  "-"}
                              </td>
                              <td className="px-4 py-3 text-slate-600">
                                {formatDate(document.uploaded_at)}
                              </td>
                              <td className="px-4 py-3 text-right">
                                <button
                                  type="button"
                                  onClick={() => void handleDeleteDocument(document)}
                                  disabled={deletingThisDocument}
                                  className="inline-flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm font-medium text-red-600 hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-60"
                                >
                                  {deletingThisDocument ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <Trash2 className="h-4 w-4" />
                                  )}
                                  {deletingThisDocument ? "Removing..." : "Remove"}
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            )}

            {activeTab === "report" && caseData && (
              <div className="space-y-6">
                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h2 className="text-lg font-semibold text-slate-900">
                    Generate Report
                  </h2>

                  <p className="mt-1 text-sm text-slate-600">
                    Create a comprehensive case report for review or sharing
                  </p>

                  <div className="mt-5 flex flex-wrap gap-3">
                    <button
                      type="button"
                      onClick={() => fetchAnalysisReports()}
                      className="rounded-xl bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-900 hover:bg-cyan-300"
                    >
                      Refresh Reports
                    </button>

                    {financialAnalysisReport && (
                      <>
                        <ReportExportButton
                          label="HTML"
                          format="html"
                          exportingFormat={exportingReportFormat}
                          onClick={downloadAnalysisReport}
                        />
                        <ReportExportButton
                          label="PDF"
                          format="pdf"
                          exportingFormat={exportingReportFormat}
                          onClick={downloadAnalysisReport}
                        />
                        <ReportExportButton
                          label="Excel"
                          format="excel"
                          exportingFormat={exportingReportFormat}
                          onClick={downloadAnalysisReport}
                        />
                      </>
                    )}
                  </div>

                  {reportExportError && (
                    <p className="mt-3 text-sm text-red-600">
                      {reportExportError}
                    </p>
                  )}
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h2 className="text-lg font-semibold text-slate-900">
                    Report Preview
                  </h2>

                  <div className="mt-6 border-b border-slate-200 pb-6">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <h1 className="text-3xl font-bold text-slate-900">
                          {caseData.company_name}
                        </h1>
                        <p className="mt-1 text-slate-600">
                          Loan Consultation Report
                        </p>
                      </div>

                      <div className="text-right text-sm text-slate-600">
                        <p>{caseData.case_code || caseData.id}</p>
                        <p>{new Date().toLocaleDateString()}</p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 grid gap-4 md:grid-cols-4">
                    <div className="rounded-xl bg-slate-100 p-4">
                      <p className="text-sm text-slate-500">Loan Amount</p>
                      <p className="mt-1 text-xl font-bold text-slate-900">
                        {formatCurrency(caseData.requested_amount)}
                      </p>
                    </div>

                    <div className="rounded-xl bg-slate-100 p-4">
                      <p className="text-sm text-slate-500">Status</p>
                      <p className="mt-1 font-semibold text-slate-900">
                        {caseData.status || "New"}
                      </p>
                    </div>

                    <div className="rounded-xl bg-slate-100 p-4">
                      <p className="text-sm text-slate-500">Industry</p>
                      <p className="mt-1 font-semibold text-slate-900">
                        {caseData.industry}
                      </p>
                    </div>

                    <div className="rounded-xl bg-red-100 p-4">
                      <p className="text-sm text-red-600">Risk Assessment</p>
                      <p className="mt-1 font-bold text-red-700">
                        Pending Analysis
                      </p>
                    </div>
                  </div>

                  <div className="mt-8">
                    <h3 className="text-lg font-semibold text-slate-900">
                      Client Information
                    </h3>

                    <div className="mt-4 grid gap-4 md:grid-cols-2">
                      <div className="border-b border-slate-200 py-3">
                        <p className="text-sm text-slate-500">Client Name</p>
                        <p className="font-medium text-slate-900">
                          {caseData.client_name}
                        </p>
                      </div>

                      <div className="border-b border-slate-200 py-3">
                        <p className="text-sm text-slate-500">Email</p>
                        <p className="font-medium text-slate-900">
                          {caseData.email}
                        </p>
                      </div>

                      <div className="border-b border-slate-200 py-3">
                        <p className="text-sm text-slate-500">Phone</p>
                        <p className="font-medium text-slate-900">
                          {caseData.phone || "-"}
                        </p>
                      </div>

                      <div className="border-b border-slate-200 py-3">
                        <p className="text-sm text-slate-500">Loan Purpose</p>
                        <p className="font-medium text-slate-900">
                          {caseData.loan_purpose || "-"}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-8">
                    <h3 className="text-lg font-semibold text-slate-900">
                      Financial Analysis Result
                    </h3>

                    {!financialAnalysisReport ? (
                      <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center">
                        <p className="font-medium text-slate-700">
                          No financial analysis report generated yet
                        </p>
                        <p className="mt-1 text-sm text-slate-500">
                          Go to Analysis tab, select financial statement documents, then run analysis.
                        </p>
                      </div>
                    ) : (
                      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
                        <div
                          className="prose max-w-none"
                          dangerouslySetInnerHTML={{
                            __html: financialAnalysisReport.report_html,
                          }}
                        />
                      </div>
                    )}
                  </div>

                  <div className="mt-8">
                    <h3 className="text-lg font-semibold text-slate-900">
                      Bank Statement Analysis Result
                    </h3>

                    {!bankAnalysisReport ? (
                      <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center">
                        <p className="font-medium text-slate-700">
                          No bank statement report generated yet
                        </p>
                        <p className="mt-1 text-sm text-slate-500">
                          Go to Analysis tab, select bank statement PDFs, then run bank analysis.
                        </p>
                      </div>
                    ) : (
                      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
                        <div
                          className="prose max-w-none"
                          dangerouslySetInnerHTML={{
                            __html: bankAnalysisReport.report_html,
                          }}
                        />
                      </div>
                    )}
                  </div>
                </section>
              </div>
            )}

            {activeTab === "analysis" && caseData && (
              <div className="space-y-6">
                <FinancialStatementAnalysisErrorBoundary>
                  <FinancialStatementAnalysisSection
                    caseId={caseData.id}
                    documents={documents}
                    onAnalysisSaved={setFinancialAnalysisReport}
                    onAnalysisReportsRefresh={fetchAnalysisReports}
                  />
                </FinancialStatementAnalysisErrorBoundary>

                <BankStatementAnalysisErrorBoundary>
                  <BankStatementAnalysisSection
                    caseId={caseData.id}
                    documents={documents}
                    onAnalysisSaved={setBankAnalysisReport}
                    onAnalysisReportsRefresh={fetchAnalysisReports}
                  />
                </BankStatementAnalysisErrorBoundary>
              </div>
            )}

            {activeTab === "notes" && (
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <h2 className="text-lg font-semibold text-slate-900">Notes</h2>

                {isEditing && caseData ? (
                  <textarea
                    value={caseData.initial_notes || ""}
                    onChange={(e) =>
                      setCaseData({
                        ...caseData,
                        initial_notes: e.target.value,
                      })
                    }
                    className="mt-3 min-h-40 w-full rounded-xl border border-slate-300 p-3 text-sm"
                  />
                ) : (
                  <p className="mt-3 text-sm text-slate-700">
                    {caseData?.initial_notes || "No notes yet."}
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}

function ReportExportButton({
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
      className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
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

function getApiErrorMessage(
  result: { error?: unknown; code?: unknown },
  fallback: string
) {
  const message = typeof result.error === "string" ? result.error : fallback;
  const code = typeof result.code === "string" ? result.code : "";

  return code ? `${message} (${code})` : message;
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
