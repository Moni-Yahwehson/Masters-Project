<template>
  <div class="anima-app-container">
    <!-- Left Navigation Sidebar -->
    <Sidebar />

    <!-- Main Chat Workspace -->
    <main class="anima-main-content">
      <!-- Top Bar Header -->
      <HeaderBar />

      <!-- Chat Body Scroll Area -->
      <div class="chat-viewport">
        <!-- If active chat is empty, show Hero Empty State -->
        <EmptyState v-if="!messages || messages.length === 0" />

        <!-- Otherwise, show list of conversation messages -->
        <div v-else class="messages-container" ref="messagesContainerRef">
          <ChatMessage 
            v-for="msg in messages" 
            :key="msg.id" 
            :message="msg" 
          />
        </div>
      </div>

      <!-- Floating Bottom Input Bar (shown when in active conversation) -->
      <div v-if="messages && messages.length > 0" class="floating-input-wrapper">
        <ChatInput />
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import Sidebar from './components/Sidebar.vue'
import HeaderBar from './components/HeaderBar.vue'
import EmptyState from './components/EmptyState.vue'
import ChatMessage from './components/ChatMessage.vue'
import ChatInput from './components/ChatInput.vue'
import { useChat } from './composables/useChat'

const { messages } = useChat()
const messagesContainerRef = ref(null)

// Auto-scroll to bottom when messages update
watch(
  () => messages.value.length,
  () => {
    nextTick(() => {
      if (messagesContainerRef.value) {
        messagesContainerRef.value.scrollTop = messagesContainerRef.value.scrollHeight
      }
    })
  }
)
</script>

<style scoped>
.anima-app-container {
  display: flex;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background-color: var(--color-bg-canvas);
}

.anima-main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100%;
  position: relative;
  overflow: hidden;
  background-color: var(--color-bg-canvas);
}

.chat-viewport {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  position: relative;
}

.messages-container {
  flex: 1;
  width: 100%;
  max-width: 760px;
  margin: 0 auto;
  padding: 32px 24px 140px; /* Padding bottom for floating input bar */
  overflow-y: auto;
}

.floating-input-wrapper {
  position: absolute;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  width: 100%;
  max-width: 660px;
  padding: 0 20px;
  z-index: 10;
}

@media (max-width: 768px) {
  .messages-container {
    padding: 16px 16px 140px;
  }

  .floating-input-wrapper {
    bottom: 16px;
    padding: 0 16px;
  }
}
</style>
