import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { getKnowledgeBases, getServiceInfo } from '../api/client'
import type { KnowledgeBaseList, ServiceInfo } from '../api/types'

interface Conversation { id: string; title: string; agent: string; createdAt: number }

function readConversations(): Conversation[] {
  // 本地只保存会话索引；消息正文仍从服务端检查点读取。
  try { return JSON.parse(localStorage.getItem('scholarflow-conversations') || '[]') as Conversation[] }
  catch { return [] }
}

export const useWorkspaceStore = defineStore('workspace', () => {
  // 页面共享的服务元数据、知识库列表和当前会话状态。
  const info = ref<ServiceInfo | null>(null)
  const knowledgeBases = ref<KnowledgeBaseList | null>(null)
  const conversations = ref<Conversation[]>(readConversations())
  const activeThreadId = ref<string>(localStorage.getItem('scholarflow-thread') || '')
  const selectedAgent = ref<string>(localStorage.getItem('scholarflow-agent') || 'auto')
  const selectedModel = ref<string>(localStorage.getItem('scholarflow-model') || '')
  // user_id 在不同线程间保持一致；thread_id 则用于恢复单个会话。
  const userId = localStorage.getItem('scholarflow-user') || crypto.randomUUID()
  localStorage.setItem('scholarflow-user', userId)

  const activeConversation = computed(() => conversations.value.find(item => item.id === activeThreadId.value))

  async function loadInfo(): Promise<void> {
    // 后端可用智能体和模型可能变化，旧的本地选择需回退到服务端默认值。
    info.value = await getServiceInfo()
    if (!info.value.agents.some(agent => agent.key === selectedAgent.value)) selectedAgent.value = info.value.default_agent
    if (!info.value.models.includes(selectedModel.value)) selectedModel.value = info.value.default_model
  }

  async function loadKnowledgeBases(): Promise<void> {
    knowledgeBases.value = await getKnowledgeBases()
  }

  function saveConversations(): void {
    localStorage.setItem('scholarflow-conversations', JSON.stringify(conversations.value))
  }

  function newConversation(): string {
    // 先生成线程 ID，首次发送时传给后端，保证前后端指向同一会话。
    const id = crypto.randomUUID()
    conversations.value.unshift({ id, title: '新研究对话', agent: selectedAgent.value, createdAt: Date.now() })
    saveConversations()
    selectConversation(id)
    return id
  }

  function selectConversation(id: string): void {
    // 切换会话时同步其本地记录的智能体选择。
    activeThreadId.value = id
    localStorage.setItem('scholarflow-thread', id)
    const item = conversations.value.find(conversation => conversation.id === id)
    if (item) setAgent(item.agent)
  }

  function setTitle(id: string, title: string): void {
    // 只在首条问题发送后命名一次，保留用户识别会话的标题。
    const item = conversations.value.find(conversation => conversation.id === id)
    if (item && item.title === '新研究对话') {
      item.title = title.slice(0, 36)
      saveConversations()
    }
  }

  function setAgent(agent: string): void {
    selectedAgent.value = agent
    localStorage.setItem('scholarflow-agent', agent)
  }

  function setModel(model: string): void {
    selectedModel.value = model
    localStorage.setItem('scholarflow-model', model)
  }

  return { info, knowledgeBases, conversations, activeThreadId, activeConversation, selectedAgent, selectedModel,
    userId, loadInfo, loadKnowledgeBases, newConversation, selectConversation, setTitle, setAgent, setModel }
})
