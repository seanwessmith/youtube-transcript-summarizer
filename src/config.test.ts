import { describe, expect, test } from "bun:test";

import {
  MODEL_PROFILES,
  getAppConfig,
  getModelProfile,
  getTranscriptLanguage,
} from "./config.ts";

describe("configuration", () => {
  test("defines distinct GPT-5.6 cost, balanced, and quality profiles", () => {
    expect(MODEL_PROFILES).toEqual({
      cheap: {
        summary: { model: "gpt-5.6-luna", reasoningEffort: "low", verbosity: "medium" },
        qa: { model: "gpt-5.6-luna", reasoningEffort: "low", verbosity: "low" },
      },
      balanced: {
        summary: { model: "gpt-5.6-terra", reasoningEffort: "low", verbosity: "medium" },
        qa: { model: "gpt-5.6-luna", reasoningEffort: "low", verbosity: "low" },
      },
      quality: {
        summary: { model: "gpt-5.6-sol", reasoningEffort: "medium", verbosity: "medium" },
        qa: { model: "gpt-5.6-luna", reasoningEffort: "low", verbosity: "low" },
      },
    });
  });

  test("normalizes profile and language values", () => {
    expect(getModelProfile(" BALANCED ")).toBe("balanced");
    expect(getModelProfile("unknown")).toBe("cheap");
    expect(getTranscriptLanguage(" sv ")).toBe("sv");
    expect(getTranscriptLanguage(" ")).toBe("en");
  });

  test("model overrides inherit the selected task settings", () => {
    const config = getAppConfig({
      MODEL_PROFILE: "quality",
      SUMMARY_MODEL: "custom-summary",
      QA_MODEL: "custom-qa",
      EMBEDDING_MODEL: "custom-embedding",
      TRANSCRIPT_LANGUAGE: "de",
    });
    expect(config.summary).toEqual({
      model: "custom-summary",
      reasoningEffort: "medium",
      verbosity: "medium",
    });
    expect(config.qa).toEqual({
      model: "custom-qa",
      reasoningEffort: "low",
      verbosity: "low",
    });
    expect(config.embeddingModel).toBe("custom-embedding");
    expect(config.transcriptLanguage).toBe("de");
  });
});
