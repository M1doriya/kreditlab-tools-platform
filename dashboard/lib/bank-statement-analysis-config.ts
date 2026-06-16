export const bankStatementAnalysisSection = {
  id: "bank-statement-analysis",
  label: "Bank Statement Analysis",
  action: "/api/run-bank-analysis",
  exportAction: "/api/export-bank-analysis",
  tool: {
    id: "bank-statement",
    analysisType: "bank_statement",
    displayName: "Bank Statement Analyzer",
    supportedExtensions: [".pdf"],
    accept: ".pdf,application/pdf",
    supportedBanks: [
      "Affin Bank",
      "Agro Bank",
      "Alliance Bank",
      "Ambank",
      "Bank Islam",
      "Bank Muamalat",
      "Bank Rakyat",
      "CIMB Bank",
      "Hong Leong",
      "Maybank",
      "Public Bank (PBB)",
      "RHB Bank",
      "OCBC Bank",
      "UOB Bank",
    ],
  },
} as const;

export type BankStatementBankName =
  (typeof bankStatementAnalysisSection.tool.supportedBanks)[number];
