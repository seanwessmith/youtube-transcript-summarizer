import { describe, expect, test } from "bun:test";

import { chunkText, countWords } from "./rag.ts";

describe("chunkText", () => {
  test("returns no chunks for blank input", () => {
    expect(chunkText(" \n\t ", { maxWords: 100, overlapWords: 10 })).toEqual([]);
  });

  test("enforces maxWords even when text has no punctuation", () => {
    const text = Array.from({ length: 7000 }, (_, index) => `w${index}`).join(" ");
    const chunks = chunkText(text, { maxWords: 1000, overlapWords: 100 });

    expect(chunks).toHaveLength(8);
    expect(Math.max(...chunks.map((chunk) => countWords(chunk.text)))).toBeLessThanOrEqual(
      1000
    );
    expect(chunks[0]).toMatchObject({ index: 0, startWord: 0, endWord: 1000 });
    expect(chunks.at(-1)).toMatchObject({ index: 7, startWord: 6300, endWord: 7000 });
  });

  test("includes real overlapped words in adjacent chunks", () => {
    const words = Array.from({ length: 25 }, (_, index) => `w${index}`);
    const chunks = chunkText(words.join(" "), { maxWords: 10, overlapWords: 3 });

    expect(chunks.map((chunk) => [chunk.startWord, chunk.endWord])).toEqual([
      [0, 10],
      [7, 17],
      [14, 24],
      [21, 25],
    ]);
    expect(chunks[1].text.split(/\s+/).slice(0, 3)).toEqual(["w7", "w8", "w9"]);
  });

  test("clamps overlap below maxWords", () => {
    const words = Array.from({ length: 8 }, (_, index) => `w${index}`);
    const chunks = chunkText(words.join(" "), { maxWords: 4, overlapWords: 99 });

    expect(chunks.map((chunk) => [chunk.startWord, chunk.endWord])).toEqual([
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
      [4, 8],
    ]);
  });
});
