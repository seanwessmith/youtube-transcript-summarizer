export const MODEL_PROFILES = {
  cheap: {
    SUMMARY_MODEL: "gpt-5.4-nano",
    QA_MODEL: "gpt-5.4-nano",
  },
  balanced: {
    SUMMARY_MODEL: "gpt-5.6-luna",
    QA_MODEL: "gpt-5.4-nano",
  },
  quality: {
    SUMMARY_MODEL: "gpt-5.6-sol",
    QA_MODEL: "gpt-5.6-luna",
  },
} as const;

export type ModelProfile = keyof typeof MODEL_PROFILES;

export interface AppConfig {
  modelProfile: ModelProfile;
  summaryModel: string;
  qaModel: string;
  embeddingModel: string;
  transcriptLanguage: string;
}

export const getModelProfile = (
  rawValue = process.env.MODEL_PROFILE
): ModelProfile => {
  const profile = rawValue?.trim().toLowerCase();
  return profile === "cheap" || profile === "balanced" || profile === "quality"
    ? profile
    : "cheap";
};

export const getTranscriptLanguage = (
  rawValue = process.env.TRANSCRIPT_LANGUAGE
): string => rawValue?.trim() || "en";

export const getAppConfig = (): AppConfig => {
  const modelProfile = getModelProfile();
  const selectedProfile = MODEL_PROFILES[modelProfile];

  return {
    modelProfile,
    summaryModel:
      process.env.SUMMARY_MODEL?.trim() || selectedProfile.SUMMARY_MODEL,
    qaModel: process.env.QA_MODEL?.trim() || selectedProfile.QA_MODEL,
    embeddingModel:
      process.env.EMBEDDING_MODEL?.trim() || "text-embedding-3-small",
    transcriptLanguage: getTranscriptLanguage(),
  };
};
