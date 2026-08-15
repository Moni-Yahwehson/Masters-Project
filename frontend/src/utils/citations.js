/**
 * Turn internal chunk ids into reader-facing citation numbers.
 *
 * The agents cite passages by corpus id — `[lsd-clinical-signs#01]` — and that
 * matters: the verification agent checks every clinical claim against those ids,
 * and dropping them would remove the grounding trail the whole design rests on.
 *
 * But the id is meaningless to a farmer, who has never seen the corpus. So the
 * ids stay in the data and are rendered as numbers keyed to the Sources list,
 * exactly like a footnote.
 *
 * Numbering follows first appearance in the text, so the reader meets 1 before
 * 2. Both the message body and the Sources list derive their order from this
 * one function, so the two can never disagree.
 */

// Corpus ids look like `document-slug#01`.
export const CITATION_PATTERN = /\[([a-z0-9][a-z0-9-]*#\d+)\]/gi

/**
 * Work out which passages the text actually cites, and in what order.
 *
 * @param {string} markdown  answer text, possibly containing `[chunk#01]`
 * @param {Array}  passages  RetrievedPassage[] available to cite
 * @returns {{ ordered: Array, numbers: Map<string, number>, hasCitations: boolean }}
 *   `ordered`  passages to list as Sources, in citation order
 *   `numbers`  chunk_id -> the number to show in the text
 *   `hasCitations` whether the text cited anything resolvable
 */
export function orderCitations(markdown, passages) {
  const available = new Map((passages || []).map((passage) => [passage.chunk_id, passage]))
  const numbers = new Map()
  const ordered = []

  for (const match of String(markdown || '').matchAll(CITATION_PATTERN)) {
    const id = match[1]
    // Skip ids the model invented: they resolve to no passage, so there is
    // nothing to number and nothing to show the reader.
    if (numbers.has(id) || !available.has(id)) continue
    numbers.set(id, ordered.length + 1)
    ordered.push(available.get(id))
  }

  if (ordered.length === 0) {
    // Nothing cited (or nothing resolvable) — fall back to everything
    // retrieved, unnumbered, which is what the panel showed before.
    return { ordered: passages || [], numbers, hasCitations: false }
  }
  return { ordered, numbers, hasCitations: true }
}

/**
 * Replace `[chunk#01]` markers in already-escaped HTML with footnote markers.
 *
 * Unresolvable ids are removed rather than shown: an id pointing at no passage
 * is noise to a reader and a grounding failure the verification agent reports
 * separately.
 */
export function renderCitationMarkers(html, numbers) {
  // Absorb any space before the marker so it sits tight against the word it
  // follows, the way a footnote does — otherwise "tests [id]." renders as
  // "tests 1 ." with a gap before the full stop.
  return String(html || '').replace(MARKER_PATTERN, (match, id) => {
    const number = numbers && numbers.get(id)
    if (!number) return ''
    return `<sup class="cite-marker" title="Source ${number}">${number}</sup>`
  })
}

const MARKER_PATTERN = /[ \t]*\[([a-z0-9][a-z0-9-]*#\d+)\]/gi
