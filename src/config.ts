export type ReasoningEffort = "low" | "medium";
export type TextVerbosity = "low" | "medium";

export interface ModelTaskProfile {
  model: string;
  reasoningEffort: ReasoningEffort;
  verbosity: TextVerbosity;
}

export const MODEL_PROFILES = {
  cheap: {
    summary: {
      model: "gpt-5.6-luna",
      reasoningEffort: "low",
      verbosity: "medium",
    },
    qa: {
      model: "gpt-5.6-luna",
      reasoningEffort: "low",
      verbosity: "low",
    },
  },
  balanced: {
    summary: {
      model: "gpt-5.6-terra",
      reasoningEffort: "low",
      verbosity: "medium",
    },
    qa: {
      model: "gpt-5.6-luna",
      reasoningEffort: "low",
      verbosity: "low",
    },
  },
  quality: {
    summary: {
      model: "gpt-5.6-sol",
      reasoningEffort: "medium",
      verbosity: "medium",
    },
    qa: {
      model: "gpt-5.6-luna",
      reasoningEffort: "low",
      verbosity: "low",
    },
  },
} as const satisfies Record<
  string,
  { summary: ModelTaskProfile; qa: ModelTaskProfile }
>;

export type ModelProfile = keyof typeof MODEL_PROFILES;

export interface AppConfig {
  modelProfile: ModelProfile;
  summary: ModelTaskProfile;
  qa: ModelTaskProfile;
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

export const getAppConfig = (
  environment: NodeJS.ProcessEnv = process.env
): AppConfig => {
  const modelProfile = getModelProfile(environment.MODEL_PROFILE);
  const selectedProfile = MODEL_PROFILES[modelProfile];

  return {
    modelProfile,
    summary: {
      ...selectedProfile.summary,
      model: environment.SUMMARY_MODEL?.trim() || selectedProfile.summary.model,
    },
    qa: {
      ...selectedProfile.qa,
      model: environment.QA_MODEL?.trim() || selectedProfile.qa.model,
    },
    embeddingModel:
      environment.EMBEDDING_MODEL?.trim() || "text-embedding-3-small",
    transcriptLanguage: getTranscriptLanguage(environment.TRANSCRIPT_LANGUAGE),
  };
};
