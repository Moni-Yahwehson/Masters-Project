<template>
  <div class="message-wrapper" :class="message.sender">
    <!-- USER MESSAGE -->
    <template v-if="message.sender === 'user'">
      <div v-if="message.images && message.images.length > 0" class="user-images-grid">
        <div v-for="(img, idx) in message.images" :key="idx" class="image-card">
          <img
            v-if="isRenderableImage(img)"
            :src="img"
            alt="Attached thumbnail"
            class="thumb-img"
          />
          <div v-else class="thumb-placeholder"></div>
        </div>
      </div>

      <div class="user-bubble">
        <p>{{ message.text }}</p>
      </div>
    </template>

    <!-- AI MESSAGE -->
    <template v-else-if="message.sender === 'ai'">
      <div class="ai-response-container">
        <div class="ai-header">
          <div class="starburst-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M12 2V22M2 12H22M4.93 4.93L19.07 19.07M4.93 19.07L19.07 4.93M8 2L16 22M2 8L22 16M2 16L22 8M8 22L16 2"
                stroke="#0f382a"
                stroke-width="2"
                stroke-linecap="round"
              />
            </svg>
          </div>
          <span class="ai-title">{{ message.title || 'Analysis' }}</span>
          <span v-if="message.report?.degraded" class="degraded-chip" title="Some agents were unavailable">
            degraded
          </span>
        </div>

        <div class="ai-content">
          <!-- Live agent pipeline -->
          <AgentTrace
            v-if="message.trace && message.trace.length"
            :steps="message.trace"
            :total-ms="message.report?.total_ms || null"
            :default-expanded="message.pending"
          />

          <!-- Structured diagnosis -->
          <DiagnosisPanel v-if="message.report" :report="message.report" />

          <!-- Narrative answer. Keep this pair adjacent: v-else-if only binds
               to the immediately preceding v-if. -->
          <div v-if="message.markdown" v-html="formattedContent"></div>
          <div v-else-if="message.pending" class="thinking-row">
            <span class="dot"></span><span class="dot"></span><span class="dot"></span>
          </div>

          <!-- Labelled example photographs, for questions about appearance -->
          <ReferenceGallery
            v-if="message.examples && message.examples.length"
            :examples="message.examples"
            :note="message.examplesNote"
          />

          <!-- Knowledge-base citations (text-only answers). Numbered to match
               the markers in the text above. -->
          <div
            v-if="!message.report && sourceList.length"
            class="inline-citations"
          >
            <span class="citations-label">Sources</span>
            <span
              v-for="(passage, index) in sourceList"
              :key="passage.chunk_id"
              class="citation-chip"
            >
              <span v-if="citations.hasCitations" class="chip-number">{{ index + 1 }}</span>
              {{ passage.document_title }} — {{ passage.section }}
            </span>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import AgentTrace from './AgentTrace.vue'
import DiagnosisPanel from './DiagnosisPanel.vue'
import ReferenceGallery from './ReferenceGallery.vue'
import { orderCitations, renderCitationMarkers } from '../utils/citations'

const props = defineProps({
  message: { type: Object, required: true },
})

function isRenderableImage(src) {
  return (
    typeof src === 'string' &&
    (src.startsWith('data:image/') || src.startsWith('http') || src.startsWith('/'))
  )
}

/**
 * Minimal markdown renderer.
 *
 * Model output is untrusted input as far as the DOM is concerned, so HTML is
 * escaped *before* any markdown transformation runs — the only tags that reach
 * v-html are the ones generated here.
 */
function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function renderInline(text, citationNumbers) {
  return renderCitationMarkers(
    text
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, '$1<em>$2</em>')
      .replace(/(^|[\s(])_([^_\n]+)_(?=[\s).,;:!?]|$)/g, '$1<em>$2</em>')
      // Links first: a markdown link is also bracketed, and citation markers
      // must not eat the label of one.
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'),
    citationNumbers,
  )
}

// Chunk ids are an internal grounding trail, not something a farmer can use.
// They render as footnote numbers keyed to the Sources list below.
const citations = computed(() =>
  orderCitations(props.message.markdown, props.message.citations || []),
)

const sourceList = computed(() => citations.value.ordered)

const formattedContent = computed(() => {
  const source = props.message.markdown
  if (!source) return ''

  const { numbers } = citations.value
  const lines = escapeHtml(source).split('\n')
  const html = []
  let listType = null // 'ul' | 'ol'

  const closeList = () => {
    if (listType) {
      html.push(`</${listType}>`)
      listType = null
    }
  }

  const openList = (type) => {
    if (listType !== type) {
      closeList()
      html.push(`<${type} class="ai-${type === 'ul' ? 'bullet' : 'ordered'}-list">`)
      listType = type
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.trim()

    if (!line) {
      closeList()
      continue
    }

    // Horizontal rule
    if (/^(---|___|\*\*\*)$/.test(line)) {
      closeList()
      html.push('<hr class="ai-rule" />')
      continue
    }

    // Headings (#, ##, ###) all render at one visual level
    const heading = line.match(/^(#{1,6})\s+(.*)$/)
    if (heading) {
      closeList()
      html.push(`<h4 class="ai-heading">${renderInline(heading[2], numbers)}</h4>`)
      continue
    }

    // Blockquote — used for the standing disclaimer
    if (line.startsWith('&gt; ')) {
      closeList()
      html.push(`<blockquote class="ai-quote">${renderInline(line.slice(5), numbers)}</blockquote>`)
      continue
    }

    // Ordered list
    const ordered = line.match(/^\d+[.)]\s+(.*)$/)
    if (ordered) {
      openList('ol')
      html.push(`<li>${renderInline(ordered[1], numbers)}</li>`)
      continue
    }

    // Bullet list (-, *, •)
    const bullet = line.match(/^[-*•]\s+(.*)$/)
    if (bullet) {
      openList('ul')
      html.push(`<li>${renderInline(bullet[1], numbers)}</li>`)
      continue
    }

    closeList()

    // A whole-line italic paragraph is the disclaimer convention from the API
    const emphasised = line.match(/^\*(.+)\*$/)
    if (emphasised) {
      html.push(`<p class="ai-disclaimer">${renderInline(emphasised[1], numbers)}</p>`)
      continue
    }

    html.push(`<p class="ai-paragraph">${renderInline(line, numbers)}</p>`)
  }

  closeList()
  return html.join('')
})
</script>

<style scoped>
.message-wrapper {
  width: 100%;
  display: flex;
  flex-direction: column;
  margin-bottom: 32px;
}

.message-wrapper.user { align-items: flex-end; }
.message-wrapper.ai   { align-items: flex-start; }

/* User attached images */
.user-images-grid {
  display: flex;
  gap: 10px;
  margin-bottom: 12px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.image-card {
  width: 72px;
  height: 72px;
  border-radius: 14px;
  overflow: hidden;
  background-color: #495751;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.thumb-img { width: 100%; height: 100%; object-fit: cover; }

.thumb-placeholder {
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, #44544d 0%, #34423b 100%);
}

/* User bubble */
.user-bubble {
  background-color: var(--color-primary-dark);
  color: #ffffff;
  padding: 14px 22px;
  border-radius: 24px;
  max-width: 80%;
  font-size: 14.5px;
  line-height: 1.45;
  box-shadow: 0 4px 14px rgba(15, 56, 42, 0.15);
}

.user-bubble p { margin: 0; }

/* AI response */
.ai-response-container { width: 100%; max-width: 680px; }

.ai-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.starburst-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  flex-shrink: 0;
}

.ai-title {
  font-size: 15px;
  font-weight: 600;
  color: #2b332e;
}

.degraded-chip {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  background: #fdf3e0;
  color: #8a5a12;
}

.ai-content {
  font-size: 14.5px;
  line-height: 1.6;
  color: #2c332e;
  padding-left: 40px;
}

/* Thinking dots */
.thinking-row { display: flex; gap: 5px; padding: 6px 0; }

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--color-text-placeholder);
  animation: bounce 1.2s infinite ease-in-out;
}

.dot:nth-child(2) { animation-delay: 0.15s; }
.dot:nth-child(3) { animation-delay: 0.3s; }

@keyframes bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
  30% { transform: translateY(-4px); opacity: 1; }
}

/* Inline citations for text-only answers */
.inline-citations {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border);
}

.citations-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-text-muted);
}

.citation-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11.5px;
  padding: 3px 9px;
  border-radius: 999px;
  background: var(--color-bg-sidebar);
  color: var(--color-text-muted);
}

.chip-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 15px;
  height: 15px;
  border-radius: 999px;
  background: var(--color-primary-dark);
  color: #ffffff;
  font-size: 9.5px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

/* Footnote markers standing in for corpus chunk ids. Deliberately plain: a
   pill badge reads as a button and pulls the eye away from the sentence. */
:deep(.cite-marker) {
  margin-left: 1px;
  color: var(--color-primary-dark);
  font-size: 9.5px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  vertical-align: super;
  line-height: 0;
  cursor: default;
}

/* Rendered markdown */
:deep(.ai-paragraph) { margin-bottom: 14px; }

:deep(.ai-heading) {
  font-size: 14.5px;
  font-weight: 700;
  margin: 18px 0 8px;
  color: #1a1a1a;
}

:deep(.ai-bullet-list) {
  margin: 12px 0 16px 0;
  padding-left: 20px;
  list-style-type: disc;
}

:deep(.ai-ordered-list) {
  margin: 12px 0 16px 0;
  padding-left: 20px;
  list-style-type: decimal;
}

:deep(.ai-bullet-list li),
:deep(.ai-ordered-list li) {
  margin-bottom: 6px;
  line-height: 1.5;
}

:deep(.ai-rule) {
  border: none;
  border-top: 1px solid var(--color-border);
  margin: 18px 0 14px;
}

:deep(.ai-quote) {
  border-left: 3px solid var(--color-border);
  padding-left: 12px;
  color: var(--color-text-muted);
  margin-bottom: 14px;
}

:deep(.ai-disclaimer) {
  font-size: 12.5px;
  line-height: 1.55;
  color: var(--color-text-muted);
  font-style: italic;
  margin-bottom: 10px;
}

:deep(strong) { font-weight: 700; color: #111111; }

:deep(code) {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12.5px;
  background: var(--color-bg-sidebar);
  padding: 1px 5px;
  border-radius: 4px;
}

:deep(a) { color: var(--color-primary-dark); text-decoration: underline; }

@media (max-width: 768px) {
  .ai-content { padding-left: 0; margin-top: 8px; }
  .user-bubble { max-width: 90%; font-size: 14px; padding: 12px 18px; }
  .image-card { width: 60px; height: 60px; border-radius: 10px; }
}
</style>
