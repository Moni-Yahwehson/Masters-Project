/**
 * Client for the Anima LSD diagnostic API.
 *
 * Endpoint contract (mirrors backend/app/schemas.py):
 *
 *   GET    /api/health                    -> HealthResponse
 *   GET    /api/config                    -> runtime config
 *   POST   /api/chat                      -> ChatResponse         { message, images[], session_id }
 *   GET    /api/chat/sessions             -> SessionSummary[]
 *   GET    /api/chat/sessions/{id}        -> SessionMessage[]
 *   DELETE /api/chat/sessions/{id}        -> { deleted }
 *   POST   /api/diagnose/json             -> DiagnosisReport      { message, images[] }
 *   POST   /api/diagnose/stream           -> SSE: agent_step / report / error / done
 *   GET    /api/knowledge/documents       -> DocumentInfo[]
 *   GET    /api/knowledge/search?q=&top_k -> RetrievedPassage[]
 *
 * In dev, VITE_API_BASE_URL is empty and Vite proxies /api to the backend
 * (see vite.config.js). In production, point it at the deployed API.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

function url(path) {
  return `${BASE_URL}${path}`
}

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function parseError(response) {
  let detail = null
  try {
    const body = await response.json()
    detail = body?.detail ?? body
  } catch {
    detail = await response.text().catch(() => null)
  }
  const message =
    typeof detail === 'string'
      ? detail
      : detail?.[0]?.msg || `Request failed (${response.status})`
  return new ApiError(message, response.status, detail)
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(url(path), {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    })
  } catch (cause) {
    throw new ApiError(
      'Could not reach the diagnostic service. Is the backend running on port 8000?',
      0,
      String(cause),
    )
  }
  if (!response.ok) throw await parseError(response)
  if (response.status === 204) return null
  return response.json()
}

// -- meta -------------------------------------------------------------------

export function getHealth() {
  return request('/api/health')
}

export function getConfig() {
  return request('/api/config')
}

// -- chat -------------------------------------------------------------------

/**
 * @param {string} message
 * @param {string[]} dataUrls base64 data URLs produced by FileReader
 * @param {string|null} sessionId
 */
export function sendChat(message, dataUrls = [], sessionId = null) {
  return request('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      session_id: sessionId,
      images: dataUrls.map((dataUrl) => ({ data_url: dataUrl })),
    }),
  })
}

export function listSessions() {
  return request('/api/chat/sessions')
}

export function getSession(sessionId) {
  return request(`/api/chat/sessions/${encodeURIComponent(sessionId)}`)
}

export function deleteSession(sessionId) {
  return request(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
}

// -- knowledge --------------------------------------------------------------

export function listDocuments() {
  return request('/api/knowledge/documents')
}

export function searchKnowledge(query, topK = 6) {
  const params = new URLSearchParams({ q: query, top_k: String(topK) })
  return request(`/api/knowledge/search?${params}`)
}

// -- diagnosis with live agent progress -------------------------------------

/**
 * Run the diagnosis pipeline and stream per-agent progress.
 *
 * The browser's EventSource cannot POST, so this reads the SSE body off a
 * fetch stream and parses the frames by hand.
 *
 * @param {{ message: string, dataUrls: string[], sessionId?: string|null }} input
 * @param {{ onStep?: (step) => void, onReport?: (report) => void, signal?: AbortSignal }} handlers
 * @returns {Promise<object>} the final DiagnosisReport
 */
export async function streamDiagnosis(
  { message, dataUrls, sessionId = null },
  { onStep, onReport, signal } = {},
) {
  let response
  try {
    response = await fetch(url('/api/diagnose/stream'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      signal,
      body: JSON.stringify({
        message,
        session_id: sessionId,
        images: dataUrls.map((dataUrl) => ({ data_url: dataUrl })),
      }),
    })
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause
    throw new ApiError(
      'Could not reach the diagnostic service. Is the backend running on port 8000?',
      0,
      String(cause),
    )
  }

  if (!response.ok) throw await parseError(response)
  if (!response.body) throw new ApiError('Streaming is not supported here', 0, null)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let report = null
  let streamError = null

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE frames are separated by a blank line.
      let boundary
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)

        let event = 'message'
        const dataLines = []
        for (const line of frame.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
        }
        if (!dataLines.length) continue

        let payload
        try {
          payload = JSON.parse(dataLines.join('\n'))
        } catch {
          continue
        }

        if (event === 'agent_step') onStep?.(payload)
        else if (event === 'report') {
          report = payload
          onReport?.(payload)
        } else if (event === 'error') {
          streamError = new ApiError(
            payload.detail || 'The diagnosis pipeline failed',
            500,
            payload,
          )
        }
      }
    }
  } finally {
    reader.releaseLock()
  }

  if (streamError) throw streamError
  if (!report) throw new ApiError('The pipeline ended without returning a report', 0, null)
  return report
}

export { ApiError }
