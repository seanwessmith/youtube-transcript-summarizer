import { xai } from "@ai-sdk/xai";
import { generateText } from "ai";

const result = await generateText({
  model: xai("grok-4"),
  system: "You are Grok, a highly intelligent, helpful AI assistant.",
  prompt: "What is the meaning of life, the universe, and everything?",
});

console.log(result.text);
