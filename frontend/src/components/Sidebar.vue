<template>
  <aside 
    class="anima-sidebar"
    :class="{ 
      'is-collapsed': !isSidebarOpen,
      'is-mobile-open': isMobileMenuOpen 
    }"
  >
    <!-- Top Action Bar -->
    <div class="sidebar-header">
      <button 
        class="icon-btn toggle-btn" 
        @click="handleSidebarToggle" 
        title="Toggle Sidebar"
        aria-label="Toggle navigation menu"
      >
        <Menu :size="20" />
      </button>
    </div>

    <!-- New Chat Action -->
    <div class="new-chat-container">
      <button 
        class="new-chat-btn" 
        @click="createNewChat"
        :title="isSidebarOpen ? '' : 'New chat'"
      >
        <SquarePen :size="18" />
        <span v-if="isSidebarOpen" class="new-chat-label">New chat</span>
      </button>
    </div>

    <!-- Chats List (Visible when expanded or on mobile) -->
    <div v-if="isSidebarOpen || isMobileMenuOpen" class="chats-section">
      <div class="chats-header">Chats</div>
      
      <div class="chats-list">
        <div 
          v-for="chat in chats" 
          :key="chat.id"
          class="chat-item"
          :class="{ 'is-active': chat.id === activeChatId }"
          @click="selectChat(chat.id)"
        >
          <span class="chat-title" :title="chat.title">{{ chat.title }}</span>
          
          <!-- Options menu trigger -->
          <button 
            v-if="chat.id === activeChatId" 
            class="options-btn"
            @click.stop="toggleMenu(chat.id)"
            title="Chat options"
          >
            <MoreVertical :size="16" />
          </button>

          <!-- Context dropdown menu -->
          <div v-if="activeMenuId === chat.id" class="context-menu" @click.stop>
            <button @click="startRename(chat)" class="context-item">
              <Pencil :size="14" />
              <span>Rename</span>
            </button>
            <button @click="confirmDelete(chat.id)" class="context-item delete-item">
              <Trash2 :size="14" />
              <span>Delete</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </aside>

  <!-- Mobile Backdrop Overlay -->
  <div 
    v-if="isMobileMenuOpen" 
    class="sidebar-backdrop"
    @click="toggleMobileMenu"
  ></div>
</template>

<script setup>
import { ref } from 'vue'
import { Menu, SquarePen, MoreVertical, Pencil, Trash2 } from 'lucide-vue-next'
import { useChat } from '../composables/useChat'

const { 
  chats, 
  activeChatId, 
  isSidebarOpen, 
  isMobileMenuOpen, 
  selectChat, 
  createNewChat, 
  deleteChat, 
  renameChat, 
  toggleSidebar, 
  toggleMobileMenu 
} = useChat()

const activeMenuId = ref(null)

function handleSidebarToggle() {
  if (isMobileMenuOpen.value) {
    isMobileMenuOpen.value = false
  } else {
    toggleSidebar()
  }
}

function toggleMenu(id) {
  if (activeMenuId.value === id) {
    activeMenuId.value = null
  } else {
    activeMenuId.value = id
  }
}

function startRename(chat) {
  activeMenuId.value = null
  const newName = prompt('Enter new chat title:', chat.title)
  if (newName !== null && newName.trim() !== '') {
    renameChat(chat.id, newName)
  }
}

function confirmDelete(id) {
  activeMenuId.value = null
  deleteChat(id)
}
</script>

<style scoped>
.anima-sidebar {
  width: var(--sidebar-width-expanded);
  height: 100%;
  background-color: var(--color-bg-sidebar);
  display: flex;
  flex-direction: column;
  padding: 16px 14px;
  transition: width var(--transition-speed) ease, transform var(--transition-speed) ease;
  z-index: 100;
  flex-shrink: 0;
  border-right: 1px solid rgba(0, 0, 0, 0.04);
}

.anima-sidebar.is-collapsed {
  width: var(--sidebar-width-collapsed);
  padding: 16px 10px;
}

.sidebar-header {
  display: flex;
  align-items: center;
  margin-bottom: 24px;
  padding-left: 4px;
}

.icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 8px;
  color: var(--color-text-main);
  transition: background-color 0.2s;
}

.icon-btn:hover {
  background-color: rgba(0, 0, 0, 0.06);
}

.new-chat-container {
  margin-bottom: 28px;
}

.new-chat-btn {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  color: var(--color-text-main);
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  width: 100%;
  transition: background-color 0.2s;
}

.anima-sidebar.is-collapsed .new-chat-btn {
  justify-content: center;
  padding: 8px;
}

.new-chat-btn:hover {
  background-color: rgba(0, 0, 0, 0.06);
}

.chats-section {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.chats-header {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text-muted);
  margin-bottom: 12px;
  padding-left: 8px;
}

.chats-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.chat-item {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  border-radius: 20px;
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-main);
  cursor: pointer;
  transition: all 0.2s ease;
  user-select: none;
}

.chat-item:hover {
  background-color: var(--color-bg-sidebar-hover);
}

.chat-item.is-active {
  background-color: var(--color-primary-dark);
  color: #ffffff;
}

.chat-title {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  margin-right: 6px;
}

.options-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  opacity: 0.85;
  padding: 2px;
  border-radius: 4px;
}

.options-btn:hover {
  opacity: 1;
  background-color: rgba(255, 255, 255, 0.15);
}

.context-menu {
  position: absolute;
  top: 100%;
  right: 10px;
  background: #ffffff;
  border-radius: 10px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
  border: 1px solid var(--color-border);
  padding: 4px;
  z-index: 200;
  min-width: 120px;
}

.context-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px 12px;
  font-size: 13px;
  color: var(--color-text-main);
  border-radius: 6px;
}

.context-item:hover {
  background-color: #f2f4f0;
}

.context-item.delete-item {
  color: #d93838;
}

.context-item.delete-item:hover {
  background-color: #fdf2f2;
}

/* Mobile responsive drawer styling */
@media (max-width: 768px) {
  .anima-sidebar {
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    transform: translateX(-100%);
    width: 260px !important;
    box-shadow: 4px 0 20px rgba(0, 0, 0, 0.15);
  }

  .anima-sidebar.is-mobile-open {
    transform: translateX(0);
  }

  .sidebar-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(2px);
    z-index: 95;
  }
}
</style>
