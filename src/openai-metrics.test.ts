import { describe, expect, test } from "bun:test";

import {
  aggregateMetrics,
  estimateCostMicrousd,
  formatEstimatedCost,
  normalizeTokenUsage,
} from "./openai-metrics.ts";

describe("OpenAI request metrics", () => {
  test("normalizes response usage and cached tokens", () => {
    expect(
      normalizeTokenUsage({
        input_tokens: 10_000,
        output_tokens: 500,
        input_tokens_details: { cached_tokens: 4_000 },
      })
    ).toEqual({ inputTokens: 10_000, cachedInputTokens: 4_000, outputTokens: 500 });
  });

  test("estimates Luna cost in integer microdollars", () => {
    expect(
      estimateCostMicrousd("gpt-5.6-luna", {
        inputTokens: 10_000,
        cachedInputTokens: 4_000,
        outputTokens: 500,
      })
    ).toBe(1880);
    expect(formatEstimatedCost(1880)).toBe("$0.001880");
  });

  test("returns unavailable for arbitrary model overrides", () => {
    expect(
      estimateCostMicrousd("custom-model", {
        inputTokens: 1,
        cachedInputTokens: 0,
        outputTokens: 1,
      })
    ).toBeNull();
  });

  test("aggregates multi-request usage, duration, and known costs", () => {
    expect(
      aggregateMetrics("gpt-5.6-luna", [
        {
          model: "gpt-5.6-luna",
          inputTokens: 100,
          cachedInputTokens: 20,
          outputTokens: 10,
          durationMs: 500,
          estimatedCostMicrousd: 50,
        },
        {
          model: "gpt-5.6-luna",
          inputTokens: 200,
          cachedInputTokens: 40,
          outputTokens: 20,
          durationMs: 700,
          estimatedCostMicrousd: 100,
        },
      ])
    ).toEqual({
      model: "gpt-5.6-luna",
      inputTokens: 300,
      cachedInputTokens: 60,
      outputTokens: 30,
      durationMs: 1200,
      estimatedCostMicrousd: 150,
    });
  });
});
