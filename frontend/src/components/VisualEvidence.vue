<template>
  <div class="visual-evidence" v-if="hasAnything">
    <!-- The upload is itself a training image: the score is not a fair test. -->
    <div class="seen-warning" v-if="visual.query_in_index">
      <AlertTriangle :size="14" class="warn-icon" />
      <p>
        This photo is already in the model's training set, so it has seen it before.
        Its score here is not a fair test of the model — upload a new photograph to
        judge the system honestly.
      </p>
    </div>

    <!-- Saliency ------------------------------------------------------- -->
    <div class="evidence-card" v-if="saliency.length">
      <div class="evidence-head">
        <span class="evidence-title">Where the model looked</span>
        <span class="scale">
          <span class="scale-label">less</span>
          <span class="scale-ramp"></span>
          <span class="scale-label">more</span>
        </span>
      </div>

      <div class="saliency-list">
        <div v-for="map in saliency" :key="map.image_id" class="saliency-item">
          <div class="pair">
            <!-- Each frame is its own <figure>: a <figcaption> is only valid as
                 a direct child of one. -->
            <figure class="frame">
              <img :src="map.original_data_url" alt="The photograph as uploaded" />
              <figcaption>Photo</figcaption>
            </figure>
            <figure class="frame">
              <img
                :src="map.overlay_data_url"
                alt="The same photograph with the model's attention heat map"
              />
              <figcaption>Attention</figcaption>
            </figure>
          </div>
          <p class="saliency-caption">
            <span class="chip" :class="map.explains">
              {{ map.explains === 'lsd_like' ? 'LSD-like' : 'healthy-like' }}
            </span>
            <span>
              The warm areas are what pushed this image towards
              <strong>{{ map.explains === 'lsd_like' ? 'LSD-like' : 'healthy' }}</strong>
              ({{ (map.probability_lsd * 100).toFixed(1) }}%).
            </span>
          </p>
        </div>
      </div>

      <p class="evidence-note">
        Grad-CAM over the classifier's final convolution, computed from a 7×7 grid and
        smoothed — it points at a region, not at individual lesions. If the heat sits
        on the background rather than the animal, the score is being driven by scenery
        and should be discounted.
      </p>
    </div>

    <!-- Precedents ------------------------------------------------------ -->
    <div class="evidence-card" v-if="neighbours.length">
      <div class="evidence-head">
        <span class="evidence-title">Most similar labelled cases</span>
        <span class="evidence-tag agreement" :class="visual.neighbour_agreement">
          {{ agreementLabel }}
        </span>
      </div>

      <p class="vote">
        <span
          v-for="entry in voteEntries"
          :key="entry.label"
          class="vote-part"
          :class="entry.label"
        >
          <strong>{{ entry.count }}</strong> {{ entry.text }}
        </span>
        <span class="vote-of">of {{ neighbours.length }} nearest images</span>
      </p>

      <ul class="neighbours">
        <li v-for="item in neighbours" :key="item.reference_id">
          <img
            :src="item.thumbnail_data_url"
            :alt="`Training image ${item.reference_id}, labelled ${item.label}`"
          />
          <span class="nb-label" :class="item.label">{{ shortLabel(item.label) }}</span>
          <span class="nb-score">{{ Math.round(item.similarity * 100) }}%</span>
        </li>
      </ul>

      <p class="evidence-note">{{ visual.note }}</p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { AlertTriangle } from 'lucide-vue-next'

const props = defineProps({
  visual: { type: Object, required: true },
})

const saliency = computed(() => props.visual?.saliency || [])
const neighbours = computed(() => props.visual?.neighbours || [])
const hasAnything = computed(
  () => props.visual?.available && (saliency.value.length || neighbours.value.length),
)

const AGREEMENT_LABELS = {
  agree: 'matches the score',
  disagree: 'contradicts the score',
  split: 'evenly split',
}

const agreementLabel = computed(
  () => AGREEMENT_LABELS[props.visual?.neighbour_agreement] || 'no verdict',
)

function shortLabel(label) {
  if (label === 'lumpycows') return 'LSD'
  if (label === 'healthycows') return 'healthy'
  return label
}

// Ordered so the counts read in a stable sequence rather than object order.
const voteEntries = computed(() => {
  const counts = props.visual?.neighbour_label_counts || {}
  return Object.keys(counts)
    .sort()
    .map((label) => ({
      label,
      count: counts[label],
      text: label === 'lumpycows' ? 'LSD-affected' : shortLabel(label),
    }))
})
</script>

<style scoped>
.visual-evidence {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* Card chrome mirrors DiagnosisPanel so the two read as one panel. */
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
  margin-bottom: 10px;
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
  white-space: nowrap;
}

.evidence-tag.agreement.agree    { background: #e6f4ea; color: #1e6b3f; }
.evidence-tag.agreement.disagree { background: #fdecea; color: #a02a1e; }
.evidence-tag.agreement.split    { background: #fdf3e0; color: #8a5a12; }

.evidence-note {
  font-size: 12px;
  line-height: 1.5;
  color: var(--color-text-muted);
  margin-top: 10px;
}

/* Already-seen warning */
.seen-warning {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  border: 1px solid #f0d9a8;
  background: #fdf3e0;
  border-radius: 12px;
  padding: 10px 12px;
}

.seen-warning p {
  font-size: 12.5px;
  line-height: 1.5;
  color: #8a5a12;
  margin: 0;
}

.warn-icon {
  color: #8a5a12;
  flex-shrink: 0;
  margin-top: 2px;
}

/* Heat scale legend */
.scale {
  display: flex;
  align-items: center;
  gap: 6px;
}

.scale-label {
  font-size: 10.5px;
  color: var(--color-text-muted);
}

.scale-ramp {
  width: 56px;
  height: 6px;
  border-radius: 999px;
  /* Matches the ramp in backend/app/agents/visual_explain.py */
  background: linear-gradient(90deg, #ffe082 0%, #ffa032 45%, #eb462d 75%, #af143c 100%);
}

/* Saliency pairs */
.saliency-list {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.saliency-item { margin: 0; }

.pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.frame {
  position: relative;
  margin: 0;                 /* <figure> carries a default margin */
  border-radius: 10px;
  overflow: hidden;
  background: var(--color-bg-sidebar);
  line-height: 0;
}

/* No cropping and no forced ratio: both frames come from the same source so
   they align naturally, and cropping could hide the very thing the heat map is
   pointing at. */
.frame img {
  width: 100%;
  height: auto;
  display: block;
}

.frame figcaption {
  position: absolute;
  left: 6px;
  bottom: 6px;
  font-size: 10.5px;
  line-height: 1.4;
  font-weight: 600;
  color: #ffffff;
  background: rgba(20, 26, 22, 0.68);
  padding: 2px 7px;
  border-radius: 999px;
}

.saliency-caption {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
  font-size: 12.5px;
  line-height: 1.5;
  color: var(--color-text-muted);
}

.chip {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  flex-shrink: 0;
}

.chip.lsd_like     { background: #fdecea; color: #a02a1e; }
.chip.healthy_like { background: #e6f4ea; color: #1e6b3f; }

/* Precedent vote */
.vote {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 13px;
  color: var(--color-text-main);
  margin-bottom: 10px;
}

.vote-part { white-space: nowrap; }
.vote-part.lumpycows   { color: #a02a1e; }
.vote-part.healthycows { color: #1e6b3f; }

.vote-of {
  color: var(--color-text-muted);
  font-size: 12px;
}

.neighbours {
  list-style: none;
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 6px;
  margin: 0;
  padding: 0;
}

.neighbours li {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  background: var(--color-bg-sidebar);
  line-height: 0;
}

/* `contain`, not `cover`: these are evidence, and a square crop of an already
   tight lesion close-up hides most of what the reviewer is meant to compare. */
.neighbours img {
  width: 100%;
  display: block;
  aspect-ratio: 4 / 3;
  object-fit: contain;
  background: #f2f4f0;
}

.nb-label {
  position: absolute;
  top: 4px;
  left: 4px;
  font-size: 9.5px;
  line-height: 1.4;
  font-weight: 700;
  padding: 1px 5px;
  border-radius: 999px;
  letter-spacing: 0.01em;
}

.nb-label.lumpycows   { background: #fdecea; color: #a02a1e; }
.nb-label.healthycows { background: #e6f4ea; color: #1e6b3f; }

.nb-score {
  position: absolute;
  right: 4px;
  bottom: 4px;
  font-size: 9.5px;
  line-height: 1.4;
  font-variant-numeric: tabular-nums;
  color: #ffffff;
  background: rgba(20, 26, 22, 0.68);
  padding: 1px 5px;
  border-radius: 999px;
}

@media (max-width: 768px) {
  .neighbours { grid-template-columns: repeat(3, 1fr); }
}
</style>
