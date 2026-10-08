// 与 FastAPI 响应字段保持一致，避免页面组件直接依赖松散的 JSON 对象。
export interface AgentInfo { key: string; description: string }
export interface ServiceInfo {
  agents: AgentInfo[]
  models: string[]
  default_agent: string
  default_model: string
}
export interface ChatMessage {
  type: 'human' | 'ai' | 'tool' | 'custom'
  content: string
  tool_calls?: { name: string; args: Record<string, unknown> }[]
  run_id?: string | null
  custom_data?: Record<string, unknown>
}
export type StreamEvent =
  // token 用于即时展示；完整 message 到达时，页面用它替换对应的 token 草稿。
  | { type: 'message'; content: ChatMessage }
  | { type: 'token'; content: string }
  | { type: 'approval_required'; content: Record<string, unknown> }
  | { type: 'error'; content: string }

export interface KnowledgeBase {
  // 列表由服务端扫描本地知识库目录生成。
  name: string
  db_type: 'chroma' | 'qdrant'
  db_path: string
  collection_name: string | null
  source_count: number
  has_bm25: boolean
}

export interface KnowledgeBaseList {
  items: KnowledgeBase[]
  current: { db_type: string; db_path: string; collection_name: string | null }
}

export interface UploadResult {
  success: boolean
  db_name: string
  db_path?: string
  total_files: number
  total_chunks: number
  processed_files: string[]
  errors: string[]
  pageindex: unknown[]
  switched?: boolean
  switch_error?: string
}
