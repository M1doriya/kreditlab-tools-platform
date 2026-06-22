// Pricing is a manual config and must be kept updated when Anthropic pricing changes.
export const CLAUDE_EFFORT_LEVELS = [
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const;

export type ClaudeEffort = (typeof CLAUDE_EFFORT_LEVELS)[number];

export const CLAUDE_MODEL_PRICING = [
  {
    id: "claude-opus-4-8",
    label: "Claude Opus 4.8",
    inputUsdPerMillionTokens: 5,
    outputUsdPerMillionTokens: 25,
    maxOutputTokens: 128000,
    defaultOutputTokens: 64000,
    defaultEffort: "high",
  },
  {
    id: "claude-sonnet-4-6",
    label: "Claude Sonnet 4.6",
    inputUsdPerMillionTokens: 3,
    outputUsdPerMillionTokens: 15,
    maxOutputTokens: 64000,
    defaultEffort: "medium",
  },
  {
    id: "claude-haiku-4-5-20251001",
    label: "Claude Haiku 4.5",
    inputUsdPerMillionTokens: 1,
    outputUsdPerMillionTokens: 5,
    maxOutputTokens: 64000,
  },
  {
    id: "claude-opus-4-7",
    label: "Claude Opus 4.7",
    inputUsdPerMillionTokens: 5,
    outputUsdPerMillionTokens: 25,
    maxOutputTokens: 128000,
    defaultOutputTokens: 64000,
    defaultEffort: "high",
  },
] as const;

export type ClaudeModelId = (typeof CLAUDE_MODEL_PRICING)[number]["id"];

export const DEFAULT_CLAUDE_MODEL_ID: ClaudeModelId = "claude-opus-4-8";

export function getClaudeModelById(modelId: string) {
  return CLAUDE_MODEL_PRICING.find((model) => model.id === modelId) || null;
}

export function isClaudeEffort(value: string): value is ClaudeEffort {
  return CLAUDE_EFFORT_LEVELS.some((level) => level === value);
}

export function resolveClaudeEffort(
  modelId: string,
  envEffort?: string | null
): ClaudeEffort | null {
  const model = getClaudeModelById(modelId);
  const trimmedEnvEffort = envEffort?.trim();

  if (trimmedEnvEffort && isClaudeEffort(trimmedEnvEffort)) {
    return trimmedEnvEffort;
  }

  return model && "defaultEffort" in model ? model.defaultEffort : null;
}

export function getDefaultClaudeModelId(envModel?: string | null): ClaudeModelId {
  if (envModel && getClaudeModelById(envModel)) {
    return envModel as ClaudeModelId;
  }

  return DEFAULT_CLAUDE_MODEL_ID;
}

export function resolveClaudeModelId(
  requestedModel?: string | null,
  envDefaultModel?: string | null
): ClaudeModelId | null {
  const trimmedRequestedModel = requestedModel?.trim();

  if (trimmedRequestedModel) {
    return getClaudeModelById(trimmedRequestedModel)
      ? (trimmedRequestedModel as ClaudeModelId)
      : null;
  }

  return getDefaultClaudeModelId(envDefaultModel);
}

export function estimateClaudeInputTokens(characters: number) {
  return Math.ceil(Math.max(characters, 0) / 4);
}

export function estimateClaudeCost(input: {
  modelId: string;
  inputCharacters: number;
  outputTokens?: number;
}) {
  const model = getClaudeModelById(input.modelId);
  const inputTokens = estimateClaudeInputTokens(input.inputCharacters);
  const outputTokens =
    input.outputTokens ??
    (model && "defaultOutputTokens" in model
      ? model.defaultOutputTokens
      : model?.maxOutputTokens) ??
    0;

  if (!model) {
    return {
      inputTokens,
      outputTokens,
      inputCostUsd: 0,
      outputCostUsd: 0,
      totalCostUsd: 0,
    };
  }

  const inputCostUsd =
    (inputTokens / 1_000_000) * model.inputUsdPerMillionTokens;
  const outputCostUsd =
    (outputTokens / 1_000_000) * model.outputUsdPerMillionTokens;

  return {
    inputTokens,
    outputTokens,
    inputCostUsd,
    outputCostUsd,
    totalCostUsd: inputCostUsd + outputCostUsd,
  };
}
