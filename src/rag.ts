export interface ChunkOptions {
  maxWords: number;
  overlapWords: number;
}

export interface TextChunk {
  index: number;
  startWord: number;
  endWord: number;
  text: string;
}

export interface RagIndex {
  chunks: TextChunk[];
  embeddings: number[][];
}

export type EmbedText = (text: string) => Promise<number[]>;
export type EmbedTexts = (texts: string[]) => Promise<number[][]>;

export interface ScoredChunk {
  chunk: TextChunk;
  score: number;
}

export function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

const normalizeChunkInput = (text: string): string =>
  text.replace(/\r/g, "").replace(/\n{3,}/g, "\n\n").trim();

export function chunkText(text: string, options: ChunkOptions): TextChunk[] {
  const normalized = normalizeChunkInput(text);
  if (!normalized) return [];

  const maxWords = Math.max(1, options.maxWords);
  const overlapWords = Math.max(0, Math.min(options.overlapWords, maxWords - 1));
  const words = normalized.split(/\s+/);
  const chunks: TextChunk[] = [];

  let startWord = 0;
  let index = 0;

  while (startWord < words.length) {
    const endWord = Math.min(startWord + maxWords, words.length);
    chunks.push({
      index,
      startWord,
      endWord,
      text: words.slice(startWord, endWord).join(" "),
    });

    if (endWord >= words.length) break;

    startWord = endWord - overlapWords;
    index += 1;
  }

  return chunks;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;

  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }

  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom ? dot / denom : 0;
}

export async function buildRagIndex(
  text: string,
  embedTexts: EmbedTexts,
  options: ChunkOptions
): Promise<RagIndex> {
  const chunks = chunkText(text, options);
  if (chunks.length === 0) {
    return { chunks, embeddings: [] };
  }

  const embeddings = await embedTexts(chunks.map((chunk) => chunk.text));
  if (embeddings.length !== chunks.length) {
    throw new Error(
      `Expected ${chunks.length} embeddings, received ${embeddings.length}.`
    );
  }

  return { chunks, embeddings };
}

export async function selectRelevantChunks(
  index: RagIndex,
  question: string,
  embedText: EmbedText,
  maxChunks: number,
  minScore = Number.NEGATIVE_INFINITY
): Promise<ScoredChunk[]> {
  if (index.chunks.length === 0 || maxChunks <= 0) return [];

  const questionEmbedding = await embedText(question);
  if (questionEmbedding.length === 0) {
    if (minScore > 0) return [];
    return index.chunks.slice(0, maxChunks).map((chunk) => ({ chunk, score: 0 }));
  }

  return index.chunks
    .map((chunk, chunkIndex) => ({
      chunk,
      score: cosineSimilarity(questionEmbedding, index.embeddings[chunkIndex] ?? []),
    }))
    .filter((entry) => entry.score >= minScore)
    .sort((a, b) => b.score - a.score || a.chunk.index - b.chunk.index)
    .slice(0, Math.min(maxChunks, index.chunks.length))
    .sort((a, b) => a.chunk.index - b.chunk.index)
    .map(({ chunk, score }) => ({ chunk, score }));
}

export function formatContext(chunks: Array<TextChunk | ScoredChunk>): string {
  return chunks
    .map((entry, index) => {
      const chunk = "chunk" in entry ? entry.chunk : entry;
      return `Excerpt ${index + 1} (words ${chunk.startWord + 1}-${chunk.endWord}):\n${chunk.text}`;
    })
    .join("\n\n");
}
