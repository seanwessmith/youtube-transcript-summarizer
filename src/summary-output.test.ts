import { afterEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import {
  buildSummaryHtml,
  exportSummaryHtml,
  formatSummaryForTerminal,
  writeSummaryPreview,
  type SummaryDocument,
} from "./summary-output.ts";

const temporaryPaths: string[] = [];
const document: SummaryDocument = {
  contentId: "youtube:test",
  title: "Odyssey <script>",
  sourceUrl: "https://www.youtube.com/watch?v=test",
  summary: `## Main Points
- A *long* journey
  - A nested detail

## Exact Quotes
- <unsafe>

[Unsafe link](javascript:alert(1))`,
};

afterEach(() => {
  for (const temporaryPath of temporaryPaths.splice(0)) {
    fs.rmSync(temporaryPath, { recursive: true, force: true });
  }
});

describe("buildSummaryHtml", () => {
  test("renders safe semantic HTML with source navigation", () => {
    const html = buildSummaryHtml(document);

    expect(html).toContain("Content-Security-Policy");
    expect(html).toContain("img-src data:");
    expect(html).toContain('<link rel="icon" href="data:image/svg+xml,');
    expect(html).toContain("font-family: ui-serif, Georgia");
    expect(html).toContain("overflow-wrap: anywhere");
    expect(html).toContain("<h2>Main Points</h2>");
    expect(html).toContain("<ul>");
    expect(html).toContain("<li>A <em>long</em> journey");
    expect(html).toContain("<blockquote>");
    expect(html).toContain("&lt;unsafe&gt;");
    expect(html).toContain('href="https://www.youtube.com/watch?v=test"');
    expect(html).not.toContain("<script>");
    expect(html).not.toContain('href="javascript:');
  });
});

describe("summary files", () => {
  test("uses a unique temporary directory for each preview", () => {
    const first = writeSummaryPreview(document);
    const second = writeSummaryPreview(document);
    temporaryPaths.push(path.dirname(first), path.dirname(second));

    expect(first).not.toBe(second);
    expect(fs.existsSync(first)).toBe(true);
    expect(fs.existsSync(second)).toBe(true);
  });

  test("exports a durable named HTML document", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "summary-export-test-"));
    temporaryPaths.push(directory);
    const filename = exportSummaryHtml(document, directory);

    expect(path.dirname(filename)).toBe(directory);
    expect(path.basename(filename)).toMatch(/^Odyssey_script_summary_\d{4}-\d{2}-\d{2}\.html$/);
    expect(fs.existsSync(filename)).toBe(true);
  });
});

describe("formatSummaryForTerminal", () => {
  test("renders Markdown hierarchy without raw Markdown markers", () => {
    const output = formatSummaryForTerminal(
      "The Odyssey",
      `## Overall Summary
An *episodic* journey home.

## Main Points
- A long first point that needs to wrap cleanly in a narrow terminal window.
  - A nested supporting detail

## Exact Quotes
- “My name is Utis”`,
      { width: 48, useColor: false }
    );

    expect(output).toContain("The Odyssey\n────────────────────────────────────────");
    expect(output).toContain("Overall Summary\n\n  An episodic journey home.");
    expect(output).toContain("  • A long first point that needs to wrap");
    expect(output).toContain("    cleanly in a narrow terminal window.");
    expect(output).toContain("    ◦ A nested supporting detail");
    expect(output).toContain("  • “My name is Utis”");
    expect(output).not.toContain("##");
    expect(output).not.toContain("\x1b[");
  });

  test("uses ANSI styling only when requested", () => {
    const output = formatSummaryForTerminal("Title", "## Main Points\n- Detail", {
      useColor: true,
    });

    expect(output).toContain("\x1b[1;37mTitle\x1b[0m");
    expect(output).toContain("\x1b[1;36mMain Points\x1b[0m");
  });
});
