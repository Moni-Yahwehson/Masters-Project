<template>
  <header class="anima-header">
    <div class="header-left">
      <!-- Mobile hamburger trigger -->
      <button 
        class="mobile-menu-btn" 
        @click="toggleMobileMenu" 
        aria-label="Open navigation menu"
      >
        <Menu :size="22" />
      </button>

      <!-- Anima Brand Title -->
      <span class="brand-title">Anima</span>
    </div>

    <!-- Active Chat Title centered in header bar -->
    <div class="header-center">
      <h1 v-if="activeChat && activeChat.messages.length > 0" class="active-title">
        {{ activeChat.title }}
      </h1>
    </div>

    <div class="header-right">
      <!-- Reserved for potential action items or clean balance -->
    </div>
  </header>
</template>

<script setup>
import { Menu } from 'lucide-vue-next'
import { useChat } from '../composables/useChat'

const { activeChat, toggleMobileMenu } = useChat()
</script>

<style scoped>
.anima-header {
  height: 64px;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  background-color: var(--color-bg-canvas);
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.mobile-menu-btn {
  display: none;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  color: var(--color-text-main);
  border-radius: 8px;
}

.brand-title {
  font-size: 22px;
  font-weight: 700;
  color: var(--color-brand-green);
  letter-spacing: -0.3px;
}

.header-center {
  flex: 1;
  display: flex;
  justify-content: center;
  text-align: center;
  padding: 0 16px;
}

.active-title {
  font-size: 20px;
  font-weight: 600;
  color: #1a1a1a;
  letter-spacing: -0.2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 400px;
}

.header-right {
  min-width: 40px;
}

@media (max-width: 768px) {
  .anima-header {
    padding: 0 16px;
    height: 56px;
  }

  .mobile-menu-btn {
    display: flex;
  }

  .brand-title {
    font-size: 19px;
  }

  .active-title {
    font-size: 16px;
    max-width: 200px;
  }
}
</style>
