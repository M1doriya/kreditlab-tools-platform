import { financialStatementAnalysisSection } from "@/lib/financial-statement-analysis-config";
import {
  runFinancialStatementAnalysisFromText,
  type FinancialStatementJsonInput,
  type FinancialStatementTextInput,
} from "@/lib/server/financial-statement-analysis";

export type ToolIntegration = {
  sectionId: string;
  sectionLabel: string;
  analysisType: string;
  displayName: string;
  requiredEnv: string[];
  optionalEnv?: string[];
};

export const toolIntegrations = {
  financialStatement: {
    sectionId: financialStatementAnalysisSection.id,
    sectionLabel: financialStatementAnalysisSection.label,
    analysisType: financialStatementAnalysisSection.tool.analysisType,
    displayName: financialStatementAnalysisSection.tool.displayName,
    requiredEnv: ["OCR_SERVICE_URL"],
    optionalEnv: [
      "OCR_SERVICE_API_KEY",
      "OCR_SERVICE_TIMEOUT_MS",
      "FINANCIAL_OCR_SERVICE_URL",
      "ANTHROPIC_API_KEY",
      "CLAUDE_API_KEY",
      "ANTHROPIC_MODEL",
      "ANTHROPIC_EFFORT",
      "ANTHROPIC_MAX_TOKENS",
      "FINANCIAL_RENDERER_API_URL",
      "FINANCIAL_STATEMENT_ANALYSIS_API_URL",
      "FINANCIAL_ANALYSIS_API_URL",
      "FINANCIAL_RENDERER_DISABLE_DEFAULT_API",
      "FINANCIAL_RENDERER_PYTHON_BIN",
      "FINANCIAL_RENDERER_TIMEOUT_MS",
    ],
  },
} as const satisfies Record<string, ToolIntegration>;

export type ToolIntegrationId = keyof typeof toolIntegrations;

export type FinancialStatementIntegrationInput = {
  textDocuments?: FinancialStatementTextInput[];
  jsonDocuments?: FinancialStatementJsonInput[];
  model?: string;
};

export async function runToolIntegration(
  toolId: ToolIntegrationId,
  input: FinancialStatementIntegrationInput
) {
  switch (toolId) {
    case "financialStatement":
      return runFinancialStatementAnalysisFromText(input);
    default:
      throw new Error(`Unknown tool integration: ${String(toolId)}`);
  }
}
