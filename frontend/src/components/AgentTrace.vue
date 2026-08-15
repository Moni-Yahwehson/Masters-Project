<template>
  <div class="agent-trace" :class="{ collapsed: !expanded }">
    <button class="trace-header" @click="expanded = !expanded" type="button">
      <span class="trace-spinner" v-if="isRunning"></span>
      <ChevronRight v-else :size="14" class="chevron" :class="{ open: expanded }" />
      <span class="trace-title">{{ headline }}</span>
      <span class="trace-meta" v-if="totalMs">{{ formatMs(totalMs) }}</span>
    </button>

    <ol v-show="expanded" class="trace-steps">
      <li
        v-for="step in steps"
        :key="step.agent"
        class="trace-step"
        :class="step.status"
      >
        <span class="step-dot"></span>
        <div class="step-body">
          <div class="step-line">
            <span class="step-label">{{ step.label }}</span>
            <span class="step-duration" v-if="step.duration_ms">
              {{ formatMs(step.duration_ms) }}
            </span>
          </div>
          <p class="step-summary" v-if="step.summary">{{ step.summary }}</p>
          <p class="step-error" v-if="step.error">{{ step.error }}</p>
        </div>
      </li>
    </ol>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { ChevronRight } from 'lucide-vue-next'

const props = defineProps({
  steps: { type: Array, default: () => [] },
  totalMs: { type: Number, default: null },
  defaultExpanded: { type: Boolean, default: false },
})

const expanded = ref(props.defaultExpanded)

const isRunning = computed(() =>
  props.steps.some((step) => step.status === 'running'),
)

const headline = computed(() => {
  const running = props.steps.find((step) => step.status === 'running')
  if (running) return running.label + '…'
  const done = props.steps.filter((step) =>
    ['completed', 'degraded', 'failed', 'skipped'].includes(step.status),
  ).length
  return `Agent pipeline · ${done} step${done === 1 ? '' : 's'}`
})

function formatMs(ms) {
  if (ms == null) return ''
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}
</script>

<style scoped>
.agent-trace {
  border: 1px solid var(--color-border);
  border-radius: 12px;
  background: #ffffff;
  margin-bottom: 16px;
  overflow: hidden;
}

.trace-header {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 10px 14px;
  background: transparent;
  border: none;
  cursor: pointer;
  text-align: left;
  font-family: inherit;
}

.trace-header:hover {
  background: var(--color-bg-sidebar-hover);
}

.trace-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-main);
  flex: 1;
}

.trace-meta {
  font-size: 12px;
  color: var(--color-text-muted);
  font-variant-numeric: tabular-nums;
}

.chevron {
  color: var(--color-text-muted);
  transition: transform var(--transition-speed);
  flex-shrink: 0;
}

.chevron.open {
  transform: rotate(90deg);
}

.trace-spinner {
  width: 13px;
  height: 13px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-primary-dark);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  flex-shrink: 0;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.trace-steps {
  list-style: none;
  margin: 0;
  padding: 4px 14px 12px 14px;
  border-top: 1px solid var(--color-border);
}

.trace-step {
  display: flex;
  gap: 10px;
  padding: 7px 0;
}

.step-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-top: 5px;
  flex-shrink: 0;
  background: var(--color-border);
}

.trace-step.running .step-dot {
  background: var(--color-primary-dark);
  animation: pulse 1s ease-in-out infinite;
}

.trace-step.completed .step-dot { background: #2f855a; }
.trace-step.degraded .step-dot  { background: #c98a1b; }
.trace-step.failed .step-dot    { background: #c53030; }
.trace-step.skipped .step-dot   { background: #cbd3cd; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

.step-body { flex: 1; min-width: 0; }

.step-line {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
}

.step-label {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text-main);
}

.trace-step.skipped .step-label {
  color: var(--color-text-muted);
}

.step-duration {
  font-size: 11.5px;
  color: var(--color-text-muted);
  font-variant-numeric: tabular-nums;
  flex-shrink: 0;
}

.step-summary {
  font-size: 12.5px;
  color: var(--color-text-muted);
  line-height: 1.45;
  margin-top: 2px;
  word-break: break-word;
}

.step-error {
  font-size: 12.5px;
  color: #c53030;
  line-height: 1.45;
  margin-top: 2px;
  word-break: break-word;
}
</style>
