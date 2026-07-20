/**
 * RAG answer + citation parsing helpers.
 *
 * The RAG backend prepends a single SSE-style frame to its plain-text answer:
 *
 *     data: {"citations": {"1": {"id": ..., "content": ..., "score": ...,
 *                              "retrieval_method": ..., "metadata": {...}}, ...}}
 *
 *     <markdown answer with [1], [2] inline citation markers>
 *
 * `parseRagContent` strips that frame and returns the answer prose plus a list
 * of `Citation` objects. The answer is then passed through
 * `normalizeAnswerWhitespace` and `convertCitationMarkers` so the
 * `MarkdownRenderer` turns `[n]` markers into clickable `#cite-n` chips.
 */

export interface Citation {
  index: number;
  title: string;
  url?: string;
  domain?: string;
  snippet?: string;
}

export interface ParsedRagContent {
  answer: string;
  sources: Citation[];
}

interface RawCitationRecord {
  id?: string;
  content?: string;
  score?: number;
  retrieval_method?: string;
  metadata?: Record<string, unknown>;
}

const CITATION_FRAME_RE = /^data:\s*(\{[\s\S]*?\})\n\n/;

function deriveDomain(url?: string): string | undefined {
  if (!url) return undefined;
  try {
    return new URL(url).hostname;
  } catch {
    return undefined;
  }
}

export function parseRagContent(raw: string): ParsedRagContent {
  const match = CITATION_FRAME_RE.exec(raw);
  if (!match) {
    return { answer: raw, sources: [] };
  }

  let parsed: { citations?: Record<string, RawCitationRecord> } = {};
  try {
    parsed = JSON.parse(match[1]) as { citations?: Record<string, RawCitationRecord> };
  } catch {
    // Malformed frame — surface the raw text rather than throwing.
    return { answer: raw, sources: [] };
  }

  const citations = parsed.citations ?? {};
  const sources: Citation[] = Object.entries(citations)
    .map(([idx, rec]) => {
      const r = rec as RawCitationRecord;
      const meta = (r.metadata ?? {}) as Record<string, unknown>;
      const title =
        (typeof meta.filename === 'string' && meta.filename) ||
        (typeof meta.source === 'string' && meta.source) ||
        (typeof meta.title === 'string' && meta.title) ||
        r.id ||
        `Source ${idx}`;
      const url = (typeof meta.url === 'string' && meta.url) || (typeof r.id === 'string' ? r.id : undefined);
      return {
        index: Number(idx),
        title: String(title),
        url: url ? String(url) : undefined,
        domain: deriveDomain(url ? String(url) : undefined),
        snippet: typeof r.content === 'string' ? r.content : undefined,
      };
    })
    .sort((a, b) => a.index - b.index);

  const answer = raw.slice(match[0].length);
  return { answer, sources };
}

/**
 * Convert bare `[n]` markers into markdown links (`[n](#cite-n)`) so the
 * MarkdownRenderer renders them as clickable citation chips. Already-linked
 * markers (`[n](...)`) are left untouched.
 */
export function convertCitationMarkers(text: string): string {
  return text.replace(/\[(\d+)\](?!\()/g, '[$1](#cite-$1)');
}

/** Collapse excessive blank lines / trailing whitespace for cleaner rendering. */
export function normalizeAnswerWhitespace(text: string): string {
  return text
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/**
 * Parse a web-search answer that ends with a `Sources:` block, e.g.
 *
 *     <answer prose>
 *
 *     Sources:
 *     [1] **Title** (domain) - [Link](url)
 *         *snippet*
 *
 * Returns the answer prose (without the Sources block) and the extracted
 * citations. If no Sources block is present, returns the original text with an
 * empty source list.
 */
export function parseWebSearchContent(raw: string): { answer: string; sources: Citation[] } {
  const sourcesMatch = /\n[ \t]*Sources:[ \t]*\n([\s\S]*)$/i.exec(raw);
  if (!sourcesMatch) {
    return { answer: raw, sources: [] };
  }

  const answer = raw.slice(0, sourcesMatch.index).replace(/\s+$/, '');
  const block = sourcesMatch[1];
  return { answer, sources: parseWebSearchSources(block) };
}

const SOURCE_LINE_RE = /^\[(\d+)\]\s+\*\*(.+?)\*\*\s*\((.+?)\)\s*-\s*\[Link\]\((.+?)\)/;
const SNIPPET_RE = /^[ \t]*\*(.+?)\*[ \t]*$/;

function parseWebSearchSources(block: string): Citation[] {
  const sources: Citation[] = [];
  let current: Citation | null = null;

  for (const rawLine of block.split('\n')) {
    const line = rawLine.trim();
    const m = SOURCE_LINE_RE.exec(line);
    if (m) {
      if (current) sources.push(current);
      current = {
        index: Number(m[1]),
        title: m[2].trim(),
        domain: m[3].trim(),
        url: m[4].trim(),
      };
    } else if (current) {
      const snippet = SNIPPET_RE.exec(line);
      if (snippet) {
        current.snippet = snippet[1].trim();
      }
    }
  }
  if (current) sources.push(current);

  return sources;
}
