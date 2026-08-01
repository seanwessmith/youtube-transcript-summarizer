import { marked } from "marked";
import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

export interface SummaryDocument {
  contentId: string;
  title: string;
  summary: string;
  sourceUrl: string;
}

const ANSI = {
  reset: "\x1b[0m",
  boldCyan: "\x1b[1;36m",
  boldWhite: "\x1b[1;37m",
  dim: "\x1b[2m",
  italic: "\x1b[3m",
} as const;

const escapeHtml = (text: string): string =>
  text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");

const stripInlineMarkdown = (text: string): string =>
  text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "$1")
    .replace(/(?<!_)_([^_]+)_(?!_)/g, "$1")
    .trim();

const wrapText = (text: string, width: number): string[] => {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return [];

  const lines: string[] = [];
  let line = "";

  for (const word of words) {
    if (!line || line.length + word.length + 1 <= width) {
      line = line ? `${line} ${word}` : word;
    } else {
      lines.push(line);
      line = word;
    }
  }

  if (line) lines.push(line);
  return lines;
};

const style = (text: string, code: string, useColor: boolean): string =>
  useColor ? `${code}${text}${ANSI.reset}` : text;

export const formatSummaryForTerminal = (
  title: string,
  summary: string,
  options: { width?: number; useColor?: boolean } = {}
): string => {
  const width = Math.max(40, Math.min(options.width ?? 88, 120));
  const useColor = options.useColor ?? false;
  const output: string[] = [
    style(title.trim(), ANSI.boldWhite, useColor),
    style("─".repeat(Math.min(width, Math.max(40, title.trim().length))), ANSI.dim, useColor),
  ];
  let currentSection = "";

  for (const rawLine of summary.trim().split(/\r?\n/)) {
    const line = rawLine.trimEnd();
    if (!line.trim()) {
      if (output.at(-1) !== "") output.push("");
      continue;
    }

    const heading = line.match(/^#{1,6}\s+(.+)$/);
    if (heading) {
      currentSection = stripInlineMarkdown(heading[1]);
      if (output.at(-1) !== "") output.push("");
      output.push(style(currentSection, ANSI.boldCyan, useColor), "");
      continue;
    }

    const bullet = line.match(/^(\s*)[-*+]\s+(.+)$/);
    if (bullet) {
      const nesting = Math.floor(bullet[1].length / 2);
      const indent = "  ".repeat(nesting + 1);
      const marker = nesting === 0 ? "• " : "◦ ";
      const continuation = " ".repeat(marker.length);
      const availableWidth = Math.max(20, width - indent.length - marker.length);
      const wrapped = wrapText(stripInlineMarkdown(bullet[2]), availableWidth);
      const quoteSection = currentSection.toLowerCase() === "exact quotes";

      wrapped.forEach((wrappedLine, index) => {
        const renderedLine = `${indent}${index === 0 ? marker : continuation}${wrappedLine}`;
        output.push(
          quoteSection ? style(renderedLine, ANSI.italic, useColor) : renderedLine
        );
      });
      output.push("");
      continue;
    }

    for (const paragraphLine of wrapText(stripInlineMarkdown(line), width - 2)) {
      output.push(`  ${paragraphLine}`);
    }
  }

  while (output.at(-1) === "") output.pop();
  return output.join("\n");
};

const normalizeQuoteSection = (summary: string): string => {
  let inQuoteSection = false;

  return summary
    .split(/\r?\n/)
    .map((line) => {
      const heading = line.match(/^#{1,6}\s+(.+)$/);
      if (heading) {
        inQuoteSection = stripInlineMarkdown(heading[1]).toLowerCase() === "exact quotes";
        return line;
      }

      const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
      return inQuoteSection && bullet ? `> ${bullet[1]}` : line;
    })
    .join("\n");
};

const isSafeWebUrl = (value: string): boolean => {
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:";
  } catch {
    return false;
  }
};

const renderSafeMarkdown = (summary: string): string => {
  const renderer = new marked.Renderer();
  renderer.html = ({ text }) => escapeHtml(text);
  renderer.link = function ({ href, title, tokens }) {
    const label = this.parser.parseInline(tokens);
    if (!isSafeWebUrl(href)) return label;
    const titleAttribute = title ? ` title="${escapeHtml(title)}"` : "";
    return `<a href="${escapeHtml(href)}"${titleAttribute} target="_blank" rel="noopener noreferrer">${label}</a>`;
  };
  renderer.image = ({ text }) => escapeHtml(text);

  return marked.parse(normalizeQuoteSection(summary), {
    async: false,
    gfm: true,
    renderer,
  }) as string;
};

const renderSourceLink = (sourceUrl: string): string =>
  isSafeWebUrl(sourceUrl)
    ? `<a class="source" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">Watch original video <span aria-hidden="true">↗</span></a>`
    : "";

const FAVICON =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%232c6259'/%3E%3Cpath d='M25 18l24 14-24 14z' fill='%23f5f2eb'/%3E%3C/svg%3E";

export const buildSummaryHtml = (document: SummaryDocument): string => `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'; object-src 'none'">
  <link rel="icon" href="${FAVICON}" type="image/svg+xml">
  <title>${escapeHtml(document.title)}</title>
  <style>
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #f5f2eb;
      color: #282722;
      font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
      font-size: 19px;
      line-height: 1.68;
      overflow-wrap: anywhere;
      text-rendering: optimizeLegibility;
    }
    article { width: min(760px, calc(100% - 40px)); margin: 0 auto; padding: 72px 0 96px; }
    header { border-bottom: 1px solid #c9c3b6; padding-bottom: 28px; margin-bottom: 44px; }
    h1, h2 {
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: -0.025em;
      line-height: 1.15;
    }
    h1 { font-size: clamp(2.25rem, 7vw, 4.25rem); margin: 0 0 .35em; max-width: 15ch; }
    h2 { font-size: 1.2rem; margin: 2.8em 0 1em; color: #2c6259; }
    p { margin: 0 0 1em; }
    ul, ol { padding-left: 1.4em; margin: 0 0 1.4em; }
    li { padding-left: .35em; margin: 0 0 .65em; }
    li::marker { color: #97602d; }
    li > ul, li > ol { margin-top: .65em; margin-bottom: 0; }
    blockquote {
      margin: 1.25em 0;
      padding: .15em 0 .15em 1.2em;
      border-left: 3px solid #97602d;
      font-size: 1.08em;
      font-style: italic;
    }
    blockquote p { margin: 0; }
    a { color: #2c6259; text-underline-offset: .16em; }
    .source {
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: .84em;
      font-weight: 600;
    }
    code { font-family: ui-monospace, monospace; font-size: .82em; }
    @media (prefers-color-scheme: dark) {
      body { background: #181817; color: #ebe8df; }
      header { border-color: #3d3b36; }
      h2, a { color: #8bc9bc; }
      li::marker, blockquote { border-color: #d59a61; color: inherit; }
      li::marker { color: #d59a61; }
    }
    @media (max-width: 600px) {
      body { font-size: 17px; }
      article { width: min(100% - 28px, 760px); padding-top: 38px; }
    }
  </style>
</head>
<body>
  <article>
    <header>
      <h1>${escapeHtml(document.title)}</h1>
      ${renderSourceLink(document.sourceUrl)}
    </header>
    ${renderSafeMarkdown(document.summary)}
  </article>
</body>
</html>`;

const safeFilename = (title: string): string =>
  title.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 60) || "video";

export const writeSummaryPreview = (document: SummaryDocument): string => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "video-summary-"));
  const filename = path.join(directory, "summary.html");
  fs.writeFileSync(filename, buildSummaryHtml(document), "utf8");
  return filename;
};

export const exportSummaryHtml = (
  document: SummaryDocument,
  outputDirectory = process.cwd()
): string => {
  const date = new Date().toISOString().slice(0, 10);
  const filename = path.join(outputDirectory, `${safeFilename(document.title)}_summary_${date}.html`);
  fs.writeFileSync(filename, buildSummaryHtml(document), "utf8");
  return filename;
};

export const openSummaryInBrowser = (document: SummaryDocument): string => {
  const filename = writeSummaryPreview(document);
  const command =
    process.platform === "darwin" ? "open" : process.platform === "win32" ? "cmd" : "xdg-open";
  const args = process.platform === "win32" ? ["/c", "start", "", filename] : [filename];
  const result = spawnSync(command, args, { stdio: "ignore" });

  if (result.status !== 0) {
    throw new Error(`Could not open the browser. Summary saved to ${filename}`);
  }

  return filename;
};
