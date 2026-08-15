<template>
  <div class="diagnosis-panel">
    <!-- Verdict + confidence -->
    <div class="verdict-row">
      <span class="verdict-badge" :class="report.verdict">{{ verdictLabel }}</span>
      <div class="confidence">
        <span class="confidence-label">Confidence</span>
        <div class="confidence-bar">
          <div
            class="confidence-fill"
            :class="report.verdict"
            :style="{ width: `${Math.round((report.confidence || 0) * 100)}%` }"
          ></div>
        </div>
        <span class="confidence-value">{{ Math.round((report.confidence || 0) * 100) }}%</span>
      </div>
    </div>

    <!-- CNN screening -->
    <div class="evidence-card" v-if="report.cnn">
      <div class="evidence-head">
        <span class="evidence-title">CNN screening</span>
        <span class="evidence-tag" v-if="!report.cnn.available">unavailable</span>
      </div>

      <template v-if="report.cnn.available">
        <div class="probability-meter">
          <div class="meter-track">
            <div
              class="meter-fill"
              :style="{ width: `${Math.round(report.cnn.aggregate_probability_lsd * 100)}%` }"
            ></div>
            <div
              class="meter-threshold"
              :style="{ left: `${Math.round(report.cnn.threshold * 100)}%` }"
              :title="`Decision threshold ${report.cnn.threshold}`"
            ></div>
          </div>
          <span class="meter-value">
            {{ (report.cnn.aggregate_probability_lsd * 100).toFixed(1) }}%
          </span>
        </div>
        <p class="evidence-note">
          P(LSD-like) across {{ report.cnn.predictions.length }}
          image{{ report.cnn.predictions.length === 1 ? '' : 's' }},
          threshold {{ report.cnn.threshold }}.
          A high score means the image resembles the LSD training set more than the
          healthy training set — it is not a diagnosis.
        </p>
        <ul class="per-image" v-if="report.cnn.predictions.length > 1">
          <li v-for="prediction in report.cnn.predictions" :key="prediction.image_id">
            <code>{{ prediction.image_id }}</code>
            <span>{{ (prediction.probability_lsd * 100).toFixed(1) }}%</span>
            <span class="tag" :class="{ warn: !prediction.confident }">
              {{ prediction.confident ? prediction.predicted_label : 'near threshold' }}
            </span>
          </li>
        </ul>
      </template>
      <p class="evidence-note" v-else>{{ report.cnn.note }}</p>
    </div>

    <!-- Visual explanation of the CNN's call: saliency + labelled precedents -->
    <VisualEvidence v-if="report.visual" :visual="report.visual" />

    <!-- Differentials -->
    <div class="evidence-card" v-if="differentials.length">
      <div class="evidence-head">
        <span class="evidence-title">Differential diagnoses</span>
      </div>
      <ul class="differentials">
        <li v-for="item in differentials" :key="item.condition">
          <div class="diff-line">
            <span class="diff-name">{{ item.condition }}</span>
            <span class="likelihood" :class="item.likelihood">{{ item.likelihood }}</span>
          </div>
          <p class="diff-detail">{{ item.distinguishing_features }}</p>
        </li>
      </ul>
    </div>

    <!-- Recommended actions -->
    <div class="evidence-card" v-if="actions.length">
      <div class="evidence-head">
        <span class="evidence-title">Recommended next steps</span>
        <span class="evidence-tag urgency" :class="report.synthesis?.urgency">
          {{ report.synthesis?.urgency }}
        </span>
      </div>
      <ol class="actions">
        <li v-for="(action, index) in actions" :key="index">{{ action }}</li>
      </ol>
    </div>

    <!-- Sources -->
    <div class="evidence-card" v-if="citations.length">
      <button class="evidence-head as-button" type="button" @click="showSources = !showSources">
        <span class="evidence-title">
          Sources ({{ citations.length }})
        </span>
        <ChevronRight :size="14" class="chevron" :class="{ open: showSources }" />
      </button>
      <ul v-show="showSources" class="sources">
        <li v-for="(passage, index) in citations" :key="passage.chunk_id">
          <div class="source-line">
            <!-- The number matches the marker in the answer text. The corpus id
                 is kept as a tooltip: useless to a farmer, but it is the
                 grounding trail, so it stays reachable. -->
            <span v-if="numbered" class="source-number" :title="passage.chunk_id">
              {{ index + 1 }}
            </span>
            <span class="source-title">{{ passage.document_title }} — {{ passage.section }}</span>
          </div>
          <p class="source-origin">{{ passage.source }}</p>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { ChevronRight } from 'lucide-vue-next'
import VisualEvidence from './VisualEvidence.vue'
import { verdictTitle } from '../composables/useChat'
import { orderCitations } from '../utils/citations'

const props = defineProps({
  report: { type: Object, required: true },
})

const showSources = ref(false)

const verdictLabel = computed(() => verdictTitle(props.report.verdict))
const differentials = computed(() => props.report.synthesis?.differentials || [])
const actions = computed(() => props.report.synthesis?.recommended_actions || [])

// Show only the passages the answer actually cited, falling back to everything
// retrieved when it cited nothing.
//
// Ordering comes from the answer text, so these numbers line up with the
// footnote markers rendered in the message body — the two must agree, and both
// derive from the same helper.
const citationOrder = computed(() =>
  orderCitations(props.report.answer_markdown, props.report.retrieval?.passages || []),
)

const numbered = computed(() => citationOrder.value.hasCitations)

const citations = computed(() => {
  if (citationOrder.value.hasCitations) return citationOrder.value.ordered

  // The text cited nothing resolvable; fall back to the synthesis agent's own
  // citation list, then to everything retrieved.
  const passages = props.report.retrieval?.passages || []
  const cited = new Set(props.report.synthesis?.citations || [])
  if (!cited.size) return passages
  const used = passages.filter((passage) => cited.has(passage.chunk_id))
  return used.length ? used : passages
})
</script>

<style scoped>
.diagnosis-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 18px;
}

/* Verdict */
.verdict-row {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.verdict-badge {
  font-size: 12.5px;
  font-weight: 700;
  padding: 6px 12px;
  border-radius: 999px;
  letter-spacing: 0.01em;
}

.verdict-badge.lsd_likely   { background: #fdecea; color: #a02a1e; }
.verdict-badge.lsd_unlikely { background: #e6f4ea; color: #1e6b3f; }
.verdict-badge.inconclusive { background: #fdf3e0; color: #8a5a12; }
.verdict-badge.not_cattle   { background: var(--color-bg-sidebar); color: var(--color-text-muted); }

.confidence {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  min-width: 180px;
}

.confidence-label {
  font-size: 12px;
  color: var(--color-text-muted);
}

.confidence-bar {
  flex: 1;
  height: 6px;
  border-radius: 999px;
  background: var(--color-border);
  overflow: hidden;
  min-width: 60px;
}

.confidence-fill {
  height: 100%;
  border-radius: 999px;
  background: var(--color-primary-dark);
  transition: width 0.4s ease;
}

.confidence-fill.lsd_likely   { background: #c0392b; }
.confidence-fill.inconclusive { background: #c98a1b; }
.confidence-fill.lsd_unlikely { background: #2f855a; }

.confidence-value {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--color-text-main);
  font-weight: 600;
}

/* Cards */
.evidence-card {
  border: 1px solid var(--color-border);
  border-radius: 12px;
  padding: 12px 14px;
  background: #ffffff;
}

.evidence-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}

.evidence-head.as-button {
  width: 100%;
  background: none;
  border: none;
  padding: 0;
  margin: 0;
  cursor: pointer;
  font-family: inherit;
}

.evidence-title {
  font-size: 12.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-text-muted);
}

.evidence-tag {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--color-bg-sidebar);
  color: var(--color-text-muted);
  text-transform: capitalize;
}

.evidence-tag.urgency.urgent { background: #fdecea; color: #a02a1e; }
.evidence-tag.urgency.prompt { background: #fdf3e0; color: #8a5a12; }

.chevron {
  color: var(--color-text-muted);
  transition: transform var(--transition-speed);
}

.chevron.open { transform: rotate(90deg); }

/* Probability meter */
.probability-meter {
  display: flex;
  align-items: center;
  gap: 10px;
}

.meter-track {
  position: relative;
  flex: 1;
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, #e6f4ea 0%, #fdf3e0 50%, #fdecea 100%);
  overflow: visible;
}

.meter-fill {
  height: 100%;
  border-radius: 999px;
  background: rgba(15, 56, 42, 0.72);
  transition: width 0.4s ease;
}

.meter-threshold {
  position: absolute;
  top: -3px;
  width: 2px;
  height: 16px;
  background: var(--color-text-main);
  border-radius: 1px;
}

.meter-value {
  font-size: 13px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  min-width: 48px;
  text-align: right;
}

.evidence-note {
  font-size: 12.5px;
  line-height: 1.5;
  color: var(--color-text-muted);
  margin-top: 8px;
}

.per-image {
  list-style: none;
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.per-image li {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.per-image code {
  font-size: 11px;
  color: var(--color-text-muted);
}

.per-image .tag {
  margin-left: auto;
  padding: 1px 7px;
  border-radius: 999px;
  background: var(--color-bg-sidebar);
  font-size: 11px;
}

.per-image .tag.warn { background: #fdf3e0; color: #8a5a12; }

/* Differentials */
.differentials {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.diff-line {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.diff-name {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--color-text-main);
}

.likelihood {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  text-transform: capitalize;
  flex-shrink: 0;
}

.likelihood.high     { background: #fdecea; color: #a02a1e; }
.likelihood.moderate { background: #fdf3e0; color: #8a5a12; }
.likelihood.low      { background: var(--color-bg-sidebar); color: var(--color-text-muted); }

.diff-detail {
  font-size: 12.5px;
  line-height: 1.5;
  color: var(--color-text-muted);
  margin-top: 2px;
}

/* Actions */
.actions {
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.actions li {
  font-size: 13.5px;
  line-height: 1.5;
  color: var(--color-text-main);
}

/* Sources */
.sources {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--color-border);
}

.source-line {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}

.source-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 16px;
  height: 16px;
  border-radius: 999px;
  background: var(--color-primary-light);
  color: var(--color-primary-dark);
  font-size: 10px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  flex-shrink: 0;
}

.source-title {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--color-text-main);
}

.source-origin {
  font-size: 11.5px;
  color: var(--color-text-muted);
  margin-top: 2px;
  line-height: 1.4;
}

@media (max-width: 768px) {
  .verdict-row { gap: 10px; }
  .confidence { min-width: 100%; }
}
</style>
