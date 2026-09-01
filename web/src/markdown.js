// Minimal, safe Markdown renderer for advisor answers.
//
// The advisor replies in Markdown - bold values, short headings, bullet lists -
// and the chat panel used to print that verbatim, so users saw literal
// "**Wednesday**" and "### Health Guidance:". This turns the small subset the
// advisor actually emits into HTML.
//
// Safety: the text comes from a language model, so it is untrusted. Every
// character is HTML-escaped FIRST and only our own tags are added afterwards,
// which makes injection impossible by construction. No dependency is added -
// a full Markdown library would be far more than this panel needs.

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ESCAPES[c]);

// Inline formatting, applied to already-escaped text.
function inline(text) {
  const code = [];
  // Pull code spans out first so their contents are not re-formatted. The
  // placeholder is NUL-delimited because NUL cannot appear in the escaped
  // text, so it cannot collide with content - a placeholder like ' 0 ' would
  // match ordinary numbers, and AQI answers are full of those.
  let out = text.replace(/`([^`]+)`/g, (_m, body) => {
    code.push(body);
    return `\u0000${code.length - 1}\u0000`;
  });

  // [label](https://example.com) - http(s) only, so no javascript: URLs.
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_m, label, href) => `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`);

  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  // Single * or _ for emphasis, but not the bullet marker (handled per line).
  out = out.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  out = out.replace(/(^|[^_\w])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");

  return out.replace(/\u0000(\d+)\u0000/g, (_m, i) => `<code>${code[Number(i)]}</code>`);
}

/**
 * Render a Markdown string to an HTML string.
 * Handles headings, unordered and ordered lists, paragraphs, horizontal rules
 * and the inline marks above. Anything else is treated as plain text.
 */
export function renderMarkdown(source) {
  if (!source) return "";
  const lines = escapeHtml(source).replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let list = null;      // "ul" | "ol" | null
  let paragraph = [];

  const closeParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${inline(paragraph.join("<br>"))}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (list) {
      html.push(`</${list}>`);
      list = null;
    }
  };
  const openList = (kind) => {
    if (list !== kind) {
      closeList();
      html.push(`<${kind}>`);
      list = kind;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (!line.trim()) {                                  // blank: end blocks
      closeParagraph();
      closeList();
      continue;
    }

    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/);
    if (heading) {
      closeParagraph();
      closeList();
      // One visual level: the panel is narrow, so h1..h6 all render the same.
      html.push(`<h4 class="md-h">${inline(heading[2].replace(/[:\s]+$/, ""))}</h4>`);
      continue;
    }

    if (/^\s{0,3}([-*_])\s*\1\s*\1[-*_\s]*$/.test(line)) { // --- rule
      closeParagraph();
      closeList();
      html.push("<hr class='md-hr'>");
      continue;
    }

    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    if (bullet) {
      closeParagraph();
      openList("ul");
      html.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }

    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (numbered) {
      closeParagraph();
      openList("ol");
      html.push(`<li>${inline(numbered[1])}</li>`);
      continue;
    }

    closeList();
    paragraph.push(line.trim());
  }

  closeParagraph();
  closeList();
  return html.join("");
}
