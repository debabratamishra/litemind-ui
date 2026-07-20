/**
 * Parse an LLM response that may embed Generative-UI fenced blocks.
 *
 * When Generative UI is enabled, the model emits blocks like:
 *
 *     ```ui:component_name
 *     { ...json payload... }
 *     ```
 *
 * or, for HTML apps:
 *
 *     ```ui:webapp
 *     <div>...</div>
 *     ```
 *
 * `parseGenerativeUI` splits the response into ordered segments: plain `text`
 * segments (rendered as markdown) and `ui` segments (rendered as rich
 * components). JSON payloads are parsed into `data`; HTML payloads are kept
 * verbatim in `content`.
 */

export type GenerativeUISegment =
  | { type: 'text'; content: string }
  | { type: 'ui'; component: string; content: string; data?: unknown };

// Components whose payload is raw HTML rather than JSON.
const HTML_COMPONENTS = new Set(['html', 'webapp', 'iframe_app']);

const FENCE_RE = /```([^\n]*)\n([\s\S]*?)```/g;

export function parseGenerativeUI(content: string): GenerativeUISegment[] {
  const segments: GenerativeUISegment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  FENCE_RE.lastIndex = 0;
  while ((match = FENCE_RE.exec(content)) !== null) {
    const lang = match[1].trim();
    const body = match[2];

    const before = content.slice(lastIndex, match.index);
    if (before.trim().length > 0) {
      segments.push({ type: 'text', content: before });
    }

    if (lang.startsWith('ui:')) {
      const component = lang.slice('ui:'.length).trim();
      if (HTML_COMPONENTS.has(component)) {
        segments.push({ type: 'ui', component, content: body });
      } else {
        let data: unknown;
        try {
          data = JSON.parse(body);
        } catch {
          data = undefined;
        }
        if (data === undefined) {
          // Unparseable UI block — render the original fenced text.
          segments.push({ type: 'text', content: match[0] });
        } else {
          segments.push({ type: 'ui', component, content: body, data });
        }
      }
    } else {
      // Ordinary code block — keep as text so markdown renders it.
      segments.push({ type: 'text', content: match[0] });
    }

    lastIndex = FENCE_RE.lastIndex;
  }

  const after = content.slice(lastIndex);
  if (after.trim().length > 0) {
    segments.push({ type: 'text', content: after });
  }

  return segments;
}
