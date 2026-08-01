export const PRICING_EFFECTIVE_DATE = "2026-08-01";

export interface ModelTokenPricing {
  inputUsdPerMillion: number;
  cachedInputUsdPerMillion: number;
  outputUsdPerMillion: number;
}

export const MODEL_TOKEN_PRICING: Record<string, ModelTokenPricing> = {
  "gpt-5.6-luna": {
    inputUsdPerMillion: 0.2,
    cachedInputUsdPerMillion: 0.02,
    outputUsdPerMillion: 1.2,
  },
  "gpt-5.6-terra": {
    inputUsdPerMillion: 2,
    cachedInputUsdPerMillion: 0.2,
    outputUsdPerMillion: 12,
  },
  "gpt-5.6-sol": {
    inputUsdPerMillion: 5,
    cachedInputUsdPerMillion: 0.5,
    outputUsdPerMillion: 30,
  },
};

export interface TokenUsage {
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
}

export interface RequestMetrics extends TokenUsage {
  model: string;
  durationMs: number;
  estimatedCostMicrousd: number | null;
}

export interface CompletionResult {
  text: string;
  metrics: RequestMetrics;
}

const nonNegativeInteger = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.round(value))
    : 0;

export const normalizeTokenUsage = (usage: unknown): TokenUsage => {
  const value = (usage ?? {}) as {
    input_tokens?: unknown;
    output_tokens?: unknown;
    input_tokens_details?: { cached_tokens?: unknown } | null;
  };
  return {
    inputTokens: nonNegativeInteger(value.input_tokens),
    cachedInputTokens: nonNegativeInteger(
      value.input_tokens_details?.cached_tokens
    ),
    outputTokens: nonNegativeInteger(value.output_tokens),
  };
};

export const estimateCostMicrousd = (
  model: string,
  usage: TokenUsage
): number | null => {
  const pricing = MODEL_TOKEN_PRICING[model];
  if (!pricing) return null;

  const cachedTokens = Math.min(usage.inputTokens, usage.cachedInputTokens);
  const uncachedTokens = Math.max(0, usage.inputTokens - cachedTokens);
  return Math.round(
    uncachedTokens * pricing.inputUsdPerMillion +
      cachedTokens * pricing.cachedInputUsdPerMillion +
      usage.outputTokens * pricing.outputUsdPerMillion
  );
};

export const aggregateMetrics = (
  model: string,
  metrics: RequestMetrics[]
): RequestMetrics => {
  const totals = metrics.reduce<TokenUsage & { durationMs: number }>(
    (result, item) => ({
      inputTokens: result.inputTokens + item.inputTokens,
      cachedInputTokens: result.cachedInputTokens + item.cachedInputTokens,
      outputTokens: result.outputTokens + item.outputTokens,
      durationMs: result.durationMs + item.durationMs,
    }),
    { inputTokens: 0, cachedInputTokens: 0, outputTokens: 0, durationMs: 0 }
  );
  const knownCosts = metrics.map((item) => item.estimatedCostMicrousd);
  return {
    model,
    ...totals,
    estimatedCostMicrousd: knownCosts.every((cost) => cost !== null)
      ? knownCosts.reduce<number>((sum, cost) => sum + (cost ?? 0), 0)
      : null,
  };
};

export const formatEstimatedCost = (microusd: number | null): string =>
  microusd === null ? "unavailable" : `$${(microusd / 1_000_000).toFixed(6)}`;

export const formatRequestMetrics = (metrics: RequestMetrics): string => {
  const cached = metrics.cachedInputTokens
    ? ` (${metrics.cachedInputTokens.toLocaleString()} cached)`
    : "";
  return `${metrics.model}: ${metrics.inputTokens.toLocaleString()} input${cached} + ${metrics.outputTokens.toLocaleString()} output tokens | ${formatEstimatedCost(metrics.estimatedCostMicrousd)} estimated | ${(metrics.durationMs / 1000).toFixed(1)}s`;
};
