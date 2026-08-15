import { ref, computed } from 'vue'
import {
  ApiError,
  deleteSession as apiDeleteSession,
  getHealth,
  sendChat,
  streamDiagnosis,
} from '../services/api'

/**
 * Conversation state, backed by the FastAPI diagnostic service.
 *
 * Messages carry the extra fields the diagnosis UI needs:
 *   - `trace`   AgentStep[]      live pipeline progress
 *   - `report`  DiagnosisReport  verdict, CNN output, retrieval, verification
 *   - `pending` boolean          still streaming
 */

const chats = ref([])
const activeChatId = ref(null)
const isSidebarOpen = ref(true)
const isMobileMenuOpen = ref(false)
const isSending = ref(false)
const health = ref(null)

let chatCounter = 0

function newChat(title = 'New Chat') {
  chatCounter += 1
  return {
    id: `chat-${Date.now()}-${chatCounter}`,
    sessionId: null,
    title,
    createdAt: new Date(),
    messages: [],
  }
}

// Start with a single empty conversation so the empty state renders.
chats.value = [newChat()]
activeChatId.value = chats.value[0].id

export function useChat() {
  const activeChat = computed(
    () => chats.value.find((chat) => chat.id === activeChatId.value) || null,
  )
  const messages = computed(() => activeChat.value?.messages || [])

  function selectChat(id) {
    activeChatId.value = id
    isMobileMenuOpen.value = false
  }

  function createNewChat() {
    const chat = newChat()
    chats.value.unshift(chat)
    activeChatId.value = chat.id
    isMobileMenuOpen.value = false
    return chat
  }

  function deleteChat(id) {
    const index = chats.value.findIndex((chat) => chat.id === id)
    if (index === -1) return
    const [removed] = chats.value.splice(index, 1)
    if (removed.sessionId) {
      apiDeleteSession(removed.sessionId).catch(() => {
        /* server-side cleanup is best-effort */
      })
    }
    if (activeChatId.value === id) {
      if (chats.value.length > 0) activeChatId.value = chats.value[0].id
      else createNewChat()
    }
  }

  function renameChat(id, newTitle) {
    const chat = chats.value.find((item) => item.id === id)
    if (chat && newTitle.trim()) chat.title = newTitle.trim()
  }

  function toggleSidebar() {
    if (isMobileMenuOpen.value) {
      isMobileMenuOpen.value = false
    } else if (typeof window !== 'undefined' && window.innerWidth <= 768) {
      isMobileMenuOpen.value = !isMobileMenuOpen.value
    } else {
      isSidebarOpen.value = !isSidebarOpen.value
    }
  }

  function toggleMobileMenu() {
    isMobileMenuOpen.value = !isMobileMenuOpen.value
  }

  async function refreshHealth() {
    try {
      health.value = await getHealth()
    } catch (error) {
      health.value = { status: 'degraded', notes: [error.message] }
    }
    return health.value
  }

  function upsertStep(message, step) {
    const index = message.trace.findIndex((item) => item.agent === step.agent)
    if (index === -1) message.trace.push(step)
    else message.trace.splice(index, 1, step)
  }

  /**
   * Send a message. With images attached this streams the six-agent pipeline
   * so the trace renders live; without them it's a single RAG-backed call.
   */
  async function sendMessage(text, attachedImages = []) {
    if (!activeChat.value) createNewChat()
    const chat = activeChat.value
    const trimmed = (text || '').trim()
    if (!trimmed && attachedImages.length === 0) return

    chat.messages.push({
      id: `msg-${Date.now()}`,
      sender: 'user',
      text: trimmed || 'Diagnose the attached images',
      images: [...attachedImages],
    })

    if (chat.title === 'New Chat') {
      const label = trimmed || 'Image diagnosis'
      chat.title = label.length > 28 ? `${label.slice(0, 28)}…` : label
    }

    chat.messages.push({
      id: `msg-ai-${Date.now()}`,
      sender: 'ai',
      title: attachedImages.length ? 'Running diagnosis…' : 'Searching knowledge base…',
      markdown: '',
      trace: [],
      report: null,
      citations: [],
      examples: [],
      examplesNote: '',
      pending: true,
      error: null,
    })

    // Read the entry back out of the array before mutating it.
    //
    // `push` stores the *raw* object in the proxy's target. Vue only fires a
    // reactive update for writes made *through* the proxy, so holding on to the
    // object literal and assigning to it later would update the data but never
    // re-render — the placeholder would sit there "thinking" forever even
    // though the response had arrived.
    const assistant = chat.messages[chat.messages.length - 1]
    isSending.value = true

    try {
      if (attachedImages.length > 0) {
        const report = await streamDiagnosis(
          {
            message: trimmed || 'Diagnose the attached images',
            dataUrls: attachedImages,
            sessionId: chat.sessionId,
          },
          {
            onStep: (step) => upsertStep(assistant, step),
            onReport: (partial) => {
              assistant.markdown = partial.answer_markdown
            },
          },
        )
        chat.sessionId = report.session_id || chat.sessionId
        assistant.report = report
        assistant.trace = report.trace
        assistant.markdown = report.answer_markdown
        assistant.citations = report.retrieval?.passages || []
        assistant.title = verdictTitle(report.verdict)
      } else {
        const response = await sendChat(trimmed, [], chat.sessionId)
        chat.sessionId = response.session_id
        assistant.markdown = response.markdown
        assistant.citations = response.citations || []
        // Labelled reference photos, attached when the question asked to be
        // shown something rather than told.
        assistant.examples = response.examples || []
        assistant.examplesNote = response.examples_note || ''
        assistant.title = response.title
      }
    } catch (error) {
      assistant.error = error instanceof ApiError ? error.message : String(error)
      assistant.title = 'Something went wrong'
      assistant.markdown =
        `**${assistant.error}**\n\n` +
        'Check that the backend is running (`make backend`) and, for image ' +
        'diagnosis, that the local model server is up (`make serve-ollama`). ' +
        'Run `make doctor` to diagnose the setup.'
    } finally {
      assistant.pending = false
      isSending.value = false
    }
  }

  return {
    chats,
    activeChatId,
    activeChat,
    messages,
    isSidebarOpen,
    isMobileMenuOpen,
    isSending,
    health,
    selectChat,
    createNewChat,
    deleteChat,
    renameChat,
    toggleSidebar,
    toggleMobileMenu,
    refreshHealth,
    sendMessage,
  }
}

export function verdictTitle(verdict) {
  switch (verdict) {
    case 'lsd_likely':
      return 'Lumpy Skin Disease — suspected'
    case 'lsd_unlikely':
      return 'No LSD indicators found'
    case 'not_cattle':
      return 'Cannot assess these images'
    case 'inconclusive':
    default:
      return 'Inconclusive — more information needed'
  }
}
