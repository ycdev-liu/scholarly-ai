import { SSEParser } from './sse'
import type { ChatMessage, KnowledgeBaseList, ServiceInfo, StreamEvent, UploadResult } from './types'

// 访问密钥只保存在当前浏览器会话；刷新页面可恢复，关闭会话后会清除。
let token = sessionStorage.getItem('scholarflow-token') || ''

export function setToken(value: string): void {
  token = value.trim()
  if (token) sessionStorage.setItem('scholarflow-token', token)
  else sessionStorage.removeItem('scholarflow-token')
}

export function getToken(): string { return token }

function headers(extra?: HeadersInit): Headers {
  // 统一给受保护接口附加认证头，公开接口也可以复用该方法。
  const result = new Headers(extra)
  if (token) result.set('Authorization', `Bearer ${token}`)
  return result
}

async function checked(response: Response): Promise<Response> {
  // 优先展示 FastAPI 返回的 detail，非 JSON 错误再回退到 HTTP 状态。
  if (response.ok) return response
  let message = `请求失败（HTTP ${response.status}）`
  try {
    const error = await response.json() as { detail?: string; error?: string }
    message = error.detail || error.error || message
  } catch { /* 非 JSON 错误沿用 HTTP 状态。 */ }
  throw new Error(message)
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  // 普通 JSON 接口共用状态检查和认证处理。
  const response = await checked(await fetch(url, { ...init, headers: headers(init?.headers) }))
  return response.json() as Promise<T>
}

export const getServiceInfo = () => json<ServiceInfo>('/api/info')
export const getKnowledgeBases = () => json<KnowledgeBaseList>('/api/vectordb/list')
export const getHistory = (threadId: string) => json<{ messages: ChatMessage[] }>('/api/history', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ thread_id: threadId }),
})
export const getPending = (agentId: string, threadId: string) => json<{ pending: Record<string, unknown> | null }>(
  `/api/agents/${encodeURIComponent(agentId)}/pending?thread_id=${encodeURIComponent(threadId)}`,
)

export async function streamAgent(
  agentId: string,
  payload: { message: string; model?: string; thread_id: string; user_id: string; approval?: 'approve' | 'deny' },
  onEvent: (event: StreamEvent | '[DONE]') => void,
  signal?: AbortSignal,
): Promise<void> {
  // 该接口使用 POST 携带问题和线程信息，不能直接用浏览器的 EventSource。
  const response = await checked(await fetch(`/api/agents/${encodeURIComponent(agentId)}/stream`, {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ ...payload, stream_tokens: true }),
    signal,
  }))
  if (!response.body) throw new Error('服务端未返回流式响应')
  const reader = response.body.getReader()
  const parser = new SSEParser(onEvent)
  try {
    // ReadableStream 的块边界不等于 SSE 事件边界，交由解析器拼接。
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      parser.feed(value)
    }
    parser.finish()
  } finally {
    reader.releaseLock()
  }
}

export async function uploadPapers(
  files: File[], options: { dbName: string; dbType: string; chunkSize: number; chunkOverlap: number; localEmbedding: boolean },
): Promise<UploadResult> {
  // 由浏览器为 FormData 生成 multipart 边界，不手动设置 Content-Type。
  const body = new FormData()
  files.forEach(file => body.append('files', file))
  body.set('db_name', options.dbName)
  body.set('db_type', options.dbType)
  body.set('chunk_size', String(options.chunkSize))
  body.set('chunk_overlap', String(options.chunkOverlap))
  body.set('use_local_embedding', String(options.localEmbedding))
  body.set('auto_switch', 'true')
  return json<UploadResult>('/api/documents/upload', { method: 'POST', body })
}

export function switchKnowledgeBase(dbType: string, dbPath: string, collectionName?: string | null) {
  // 后端切换的是服务级知识库配置，会影响连接该服务的所有会话。
  const body = new FormData()
  body.set('db_type', dbType)
  body.set('db_path', dbPath)
  if (collectionName) body.set('collection_name', collectionName)
  return json<{ success: boolean; error?: string }>('/api/vectordb/switch', { method: 'POST', body })
}
