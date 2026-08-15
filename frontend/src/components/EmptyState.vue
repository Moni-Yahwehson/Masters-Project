<template>
  <div class="empty-state-container">
    <div class="hero-text-group">
      <span class="greeting-subtitle">Hello There!</span>
      <h2 class="hero-title">Where should we start?</h2>
    </div>

    <!-- Centered Input in empty state -->
    <div class="centered-input-wrapper">
      <ChatInput />
    </div>
  </div>
</template>

<script setup>
import ChatInput from './ChatInput.vue'
</script>

<style scoped>
.empty-state-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 24px 20px 80px;
  width: 100%;
  max-width: 720px;
  margin: 0 auto;
}

.hero-text-group {
  text-align: center;
  margin-bottom: 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.greeting-subtitle {
  font-size: 16px;
  font-weight: 600;
  color: var(--color-text-main);
  letter-spacing: -0.1px;
}

.hero-title {
  font-size: 28px;
  font-weight: 700;
  color: #111111;
  letter-spacing: -0.4px;
}

.centered-input-wrapper {
  width: 100%;
  max-width: 640px;
}

@media (max-width: 768px) {
  .empty-state-container {
    padding: 16px 16px 40px;
  }

  .greeting-subtitle {
    font-size: 15px;
  }

  .hero-title {
    font-size: 24px;
  }
}
</style>
