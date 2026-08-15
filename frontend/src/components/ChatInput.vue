<template>
  <div class="input-bar-container">
    <div class="input-box">
      <!-- Top Label -->
      <div class="input-label">Ask Anima</div>

      <!-- Attachment Previews (if any uploaded) -->
      <div v-if="attachedImages.length > 0" class="attachment-preview-row">
        <div 
          v-for="(img, index) in attachedImages" 
          :key="index" 
          class="attachment-thumb"
        >
          <img :src="img" alt="Attachment" />
          <button @click="removeAttachment(index)" class="remove-btn" title="Remove image">
            <X :size="12" />
          </button>
        </div>
      </div>

      <!-- Upload validation message -->
      <p v-if="uploadError" class="upload-error">{{ uploadError }}</p>

      <!-- Textarea Input -->
      <textarea
        ref="inputRef"
        v-model="inputText"
        class="input-textarea"
        :placeholder="isSending ? 'Running the diagnostic pipeline…' : 'Type a message or upload image...'"
        :disabled="isSending"
        rows="1"
        @keydown.enter.exact.prevent="handleSend"
        @input="autoResize"
      ></textarea>

      <!-- Bottom Actions Row -->
      <div class="input-actions">
        <!-- Plus Attachment Button -->
        <label
          class="action-btn attach-btn"
          :class="{ disabled: isSending || attachedImages.length >= MAX_IMAGES }"
          :title="attachedImages.length >= MAX_IMAGES
            ? `Maximum ${MAX_IMAGES} images`
            : 'Attach photos'"
        >
          <Plus :size="20" />
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp,image/gif"
            multiple
            class="hidden-file-input"
            :disabled="isSending || attachedImages.length >= MAX_IMAGES"
            @change="handleFileUpload"
          />
        </label>

        <!-- Send Arrow Button -->
        <button
          class="action-btn send-btn"
          :disabled="isSending || (!inputText.trim() && attachedImages.length === 0)"
          @click="handleSend"
          :title="isSending ? 'Working…' : 'Send message'"
        >
          <span v-if="isSending" class="send-spinner"></span>
          <!-- SVG arrow icon matching exact paperplane / triangle arrowhead in screenshot -->
          <svg v-else width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { Plus, X } from 'lucide-vue-next'
import { useChat } from '../composables/useChat'

const { sendMessage, isSending } = useChat()
const inputText = ref('')
const attachedImages = ref([])
const inputRef = ref(null)
const uploadError = ref('')

// Kept in step with LSD_MAX_IMAGES / LSD_MAX_IMAGE_BYTES on the backend, so the
// user gets feedback before a round trip rather than a 400 afterwards.
const MAX_IMAGES = 6
const MAX_BYTES = 8 * 1024 * 1024
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif']

function handleSend() {
  if (isSending.value) return
  const text = inputText.value.trim()
  if (!text && attachedImages.value.length === 0) return

  sendMessage(text || 'Diagnose the attached images', attachedImages.value)
  inputText.value = ''
  attachedImages.value = []
  uploadError.value = ''

  if (inputRef.value) {
    inputRef.value.style.height = 'auto'
  }
}

function handleFileUpload(event) {
  const files = Array.from(event.target.files || [])
  event.target.value = '' // allow re-selecting the same file
  if (!files.length) return

  const problems = []
  for (const file of files) {
    if (attachedImages.value.length >= MAX_IMAGES) {
      problems.push(`Only ${MAX_IMAGES} images can be attached at once.`)
      break
    }
    if (!ALLOWED_TYPES.includes(file.type)) {
      problems.push(`${file.name}: unsupported format (use JPEG, PNG, WebP or GIF).`)
      continue
    }
    if (file.size > MAX_BYTES) {
      problems.push(`${file.name}: too large (max ${MAX_BYTES / 1024 / 1024} MB).`)
      continue
    }
    const reader = new FileReader()
    reader.onload = (e) => attachedImages.value.push(e.target.result)
    reader.onerror = () => {
      uploadError.value = `${file.name}: could not be read.`
    }
    reader.readAsDataURL(file)
  }
  uploadError.value = problems.join(' ')
}

function removeAttachment(index) {
  attachedImages.value.splice(index, 1)
  uploadError.value = ''
}

function autoResize() {
  nextTick(() => {
    if (inputRef.value) {
      inputRef.value.style.height = 'auto'
      inputRef.value.style.height = Math.min(inputRef.value.scrollHeight, 120) + 'px'
    }
  })
}
</script>

<style scoped>
.input-bar-container {
  width: 100%;
  display: flex;
  justify-content: center;
  position: relative;
}

.input-box {
  width: 100%;
  background-color: var(--color-primary-dark);
  border-radius: 24px;
  padding: 16px 20px 14px;
  display: flex;
  flex-direction: column;
  box-shadow: 
    0 16px 36px -4px rgba(15, 56, 42, 0.38), 
    0 6px 16px rgba(0, 0, 0, 0.08);
  transition: box-shadow 0.3s ease;
}

.input-box:focus-within {
  box-shadow: 
    0 20px 42px -2px rgba(15, 56, 42, 0.48), 
    0 8px 20px rgba(0, 0, 0, 0.12);
}

.input-label {
  font-size: 13px;
  font-weight: 500;
  color: #8daea1;
  margin-bottom: 8px;
  user-select: none;
}

.attachment-preview-row {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}

.attachment-thumb {
  position: relative;
  width: 52px;
  height: 52px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.attachment-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.remove-btn {
  position: absolute;
  top: 2px;
  right: 2px;
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
  border-radius: 50%;
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.input-textarea {
  width: 100%;
  background: transparent;
  color: #ffffff;
  font-size: 15px;
  line-height: 1.4;
  resize: none;
  border: none;
  outline: none;
  min-height: 24px;
  max-height: 120px;
  margin-bottom: 12px;
}

.input-textarea::placeholder {
  color: rgba(255, 255, 255, 0.4);
}

.input-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  opacity: 0.85;
  transition: opacity 0.2s, transform 0.1s;
  cursor: pointer;
}

.action-btn:hover:not(:disabled) {
  opacity: 1;
  transform: scale(1.05);
}

.action-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.hidden-file-input {
  display: none;
}

.action-btn.disabled {
  opacity: 0.35;
  cursor: not-allowed;
  pointer-events: none;
}

.input-textarea:disabled {
  cursor: wait;
  opacity: 0.7;
}

.upload-error {
  font-size: 12.5px;
  line-height: 1.4;
  color: #ffb4a8;
  margin-bottom: 8px;
}

.send-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.28);
  border-top-color: #ffffff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@media (max-width: 768px) {
  .input-box {
    border-radius: 20px;
    padding: 14px 16px 12px;
  }

  .input-textarea {
    font-size: 14px;
  }
}
</style>
