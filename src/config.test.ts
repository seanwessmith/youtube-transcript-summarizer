import { describe, expect, test } from "bun:test";

import { MODEL_PROFILES, getModelProfile, getTranscriptLanguage } from "./config.ts";

describe("configuration", () => {
  test("defines the expected model IDs for every profile", () => {
    expect(MODEL_PROFILES).toEqual({
      cheap: { SUMMARY_MODEL: "gpt-5.4-nano", QA_MODEL: "gpt-5.4-nano" },
      balanced: { SUMMARY_MODEL: "gpt-5.6-luna", QA_MODEL: "gpt-5.4-nano" },
      quality: { SUMMARY_MODEL: "gpt-5.6-sol", QA_MODEL: "gpt-5.6-luna" },
    });
  });

  test("normalizes profile and language values", () => {
    expect(getModelProfile(" BALANCED ")).toBe("balanced");
    expect(getModelProfile("unknown")).toBe("cheap");
    expect(getTranscriptLanguage(" sv ")).toBe("sv");
    expect(getTranscriptLanguage(" ")).toBe("en");
  });
});
