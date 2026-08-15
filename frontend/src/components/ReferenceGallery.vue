<template>
  <figure class="reference-gallery" v-if="examples.length">
    <figcaption class="gallery-head">
      <span class="gallery-title">Reference photographs</span>
      <span class="gallery-count">{{ examples.length }} from the labelled dataset</span>
    </figcaption>

    <ul class="gallery-grid">
      <li v-for="item in examples" :key="item.reference_id">
        <img :src="item.thumbnail_data_url" :alt="item.caption" loading="lazy" />
        <span class="tag" :class="item.label">{{ shortLabel(item.label) }}</span>
      </li>
    </ul>

    <p class="gallery-note" v-if="note">{{ note }}</p>
  </figure>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  examples: { type: Array, default: () => [] },
  note: { type: String, default: '' },
})

const examples = computed(() => props.examples || [])

function shortLabel(label) {
  if (label === 'lumpycows') return 'LSD-affected'
  if (label === 'healthycows') return 'healthy'
  return label
}
</script>

<style scoped>
.reference-gallery {
  margin: 16px 0 4px;
  padding: 12px 14px;
  border: 1px solid var(--color-border);
  border-radius: 12px;
  background: #ffffff;
}

.gallery-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}

.gallery-title {
  font-size: 12.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-text-muted);
}

.gallery-count {
  font-size: 11.5px;
  color: var(--color-text-muted);
}

.gallery-grid {
  list-style: none;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin: 0;
  padding: 0;
}

.gallery-grid li {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  background: #f2f4f0;
  line-height: 0;
}

/* `contain`: these are illustrations of an appearance, so cropping the animal
   out of the frame defeats the point. */
.gallery-grid img {
  width: 100%;
  display: block;
  aspect-ratio: 4 / 3;
  object-fit: contain;
}

.tag {
  position: absolute;
  top: 5px;
  left: 5px;
  font-size: 9.5px;
  line-height: 1.4;
  font-weight: 700;
  padding: 1px 6px;
  border-radius: 999px;
}

.tag.lumpycows   { background: #fdecea; color: #a02a1e; }
.tag.healthycows { background: #e6f4ea; color: #1e6b3f; }

.gallery-note {
  font-size: 11.5px;
  line-height: 1.5;
  color: var(--color-text-muted);
  margin: 10px 0 0;
}

@media (max-width: 768px) {
  .gallery-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
