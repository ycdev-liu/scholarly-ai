<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { getHistory, getPending, streamAgent } from '../api/client'
import type { ChatMessage, StreamEvent } from '../api/types'
import { useWorkspaceStore } from '../stores/workspace'

const workspace = useWorkspaceStore()
// 已落盘的消息、正在生成的 token 草稿和工具动态分别维护，避免重复展示。
const prompt = ref('')
const messages = ref<ChatMessage[]>([])
const draft = ref('')
const activities = ref<string[]>([])
const pending = ref<Record<string, unknown> | null>(null)
const busy = ref(false)
const error = ref('')
const scroller = ref<HTMLElement | null>(null)
let controller: AbortController | null = null

const suggestions = [
  { icon: '↗', title: '检索前沿论文', description: '探索一个研究方向的近期成果', prompt: '帮我检索多模态大模型领域近期值得关注的论文，并说明主要研究方向。' },
  { icon: '▤', title: '阅读论文证据', description: '从知识库中定位关键段落', prompt: '请基于知识库中的论文，解释 Transformer 的自注意力机制，并给出对应的论文来源与页码。' },
  { icon: '✧', title: '比较研究方法', description: '跨论文梳理方法与差异', prompt: '比较知识库中不同论文采用的方法、实验设置和主要结论，并标注来源。' },
]

const visibleMessages = computed(() => messages.value.filter(message =>
  // 工具消息只在下方的“研究工具动态”中展示，聊天主体保留用户和 AI 的正文。
  (message.type === 'human' || message.type === 'ai') && message.content.trim(),
))

function renderMarkdown(content: string): string {
  // 回答可能包含 Markdown，但写入 v-html 前必须清理潜在危险的 HTML。
  return DOMPurify.sanitize(marked.parse(content, { breaks: true }) as string)
}

function scrollBottom() { nextTick(() => scroller.value?.scrollTo({ top: scroller.value.scrollHeight, behavior: 'smooth' })) }

async function loadConversation(id: string) {
  // 切换会话时停止旧流，并清空旧页面状态，防止回复写入错误的线程。
  controller?.abort()
  controller = null
  busy.value = false
  messages.value = []
  draft.value = ''
  activities.value = []
  pending.value = null
  error.value = ''
  if (!id) return
  // 刚创建的本地线程尚无服务端检查点，无需请求历史。
  if (workspace.activeConversation?.title === '新研究对话') return
  try {
    const [history, approval] = await Promise.all([
      getHistory(id), getPending(workspace.selectedAgent, id),
    ])
    // 用户可能在请求途中再次切换会话，过期结果不能覆盖当前页面。
    if (id !== workspace.activeThreadId) return
    messages.value = history.messages
    pending.value = approval.pending
    scrollBottom()
  } catch {
    // 历史接口不可用时保持页面可用；新的发送请求会独立展示错误。
  }
}

watch(() => workspace.activeThreadId, id => { void loadConversation(id) }, { immediate: true })
watch([messages, draft], scrollBottom, { deep: true })
// 离开问答页时取消网络请求，避免组件卸载后继续处理事件。
onBeforeUnmount(() => controller?.abort())

function handleEvent(event: StreamEvent | '[DONE]') {
  // 后端会混合发送 token、完整消息、工具动态和审批事件。
  if (event === '[DONE]') return
  if (event.type === 'token') {
    draft.value += event.content
  } else if (event.type === 'message') {
    const message = event.content
    if (message.type === 'ai' && message.content.trim()) {
      // 完整 AI 消息取代同一段 token 草稿，避免流式文本重复显示。
      draft.value = ''
      messages.value.push(message)
    } else if (message.type === 'tool') {
      // 工具输出可能很长，只在动态面板显示简短摘要。
      activities.value.push(message.content.slice(0, 100) || '已调用研究工具')
    } else if (message.type === 'custom' && message.custom_data?.kind === 'approval_required') {
      pending.value = message.custom_data
    }
  } else if (event.type === 'approval_required') {
    draft.value = ''
    pending.value = event.content
  } else if (event.type === 'error') {
    error.value = event.content
  }
}

async function submit(approval?: 'approve' | 'deny') {
  const content = prompt.value.trim()
  if ((!content && !approval) || busy.value) return
  const threadId = workspace.activeThreadId || workspace.newConversation()
  // 新会话先等待切换监听完成，再写入用户消息，避免被初始化逻辑清空。
  if (!workspace.activeConversation || workspace.activeConversation.title === '新研究对话') await nextTick()
  const agentId = workspace.selectedAgent
  // 审批恢复使用同一个 thread_id，不再创建一条新的用户问题。
  const approvalMessage = approval ? '' : content
  if (!approval) {
    messages.value.push({ type: 'human', content })
    workspace.setTitle(threadId, content)
    prompt.value = ''
  }
  error.value = ''
  pending.value = null
  activities.value = []
  draft.value = ''
  busy.value = true
  controller = new AbortController()
  try {
    // 保持 user_id 与 thread_id，后端才能延续跨线程身份和本轮对话状态。
    await streamAgent(agentId, {
      message: approvalMessage,
      model: workspace.selectedModel || undefined,
      thread_id: threadId,
      user_id: workspace.userId,
      approval,
    }, handleEvent, controller.signal)
  } catch (cause) {
    if ((cause as Error).name !== 'AbortError') error.value = cause instanceof Error ? cause.message : '请求失败'
  } finally {
    busy.value = false
    controller = null
    scrollBottom()
  }
}

function handleEnter(event: KeyboardEvent) {
  // Enter 发送；输入法组合字符期间不拦截按键，Shift + Enter 保留换行。
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    void submit()
  }
}
</script>

<template>
  <div class="research-page">
    <header class="topbar">
      <div class="breadcrumb"><span>工作空间</span><span class="slash">/</span><strong>研究问答</strong></div>
      <div class="topbar-right"><span class="topbar-pill"><span class="status-dot"></span> Multi-Agent 协作</span><span class="avatar">研</span></div>
    </header>

    <div ref="scroller" class="research-scroll">
      <!-- 空会话展示问题示例；有历史消息或流式草稿时切换到消息列表。 -->
      <div v-if="!visibleMessages.length && !draft" class="welcome-content">
        <div class="welcome-eyebrow"><span class="tiny-sparkle">✳</span> 为研究而构建</div>
        <h1>让每一篇论文，<br /><em>成为你的研究线索。</em></h1>
        <p class="welcome-description">从文献检索到原文证据，让多个智能体协作完成繁杂步骤。<br />提出问题，沿着可靠的来源继续探索。</p>
        <div class="suggestion-grid">
          <button v-for="item in suggestions" :key="item.title" class="suggestion-card" @click="prompt = item.prompt">
            <span class="suggestion-icon">{{ item.icon }}</span><strong>{{ item.title }}</strong><small>{{ item.description }}</small><span class="suggestion-arrow">↗</span>
          </button>
        </div>
        <div class="workflow-hint"><span>01 <strong>检索</strong></span><i></i><span>02 <strong>阅读</strong></span><i></i><span>03 <strong>溯源</strong></span></div>
      </div>

      <div v-else class="message-list" aria-live="polite">
        <div v-for="(message, index) in visibleMessages" :key="index" class="message-row" :class="message.type">
          <div class="message-avatar">{{ message.type === 'human' ? '我' : 'S' }}</div>
          <div class="message-body"><div class="message-name">{{ message.type === 'human' ? '你' : 'ScholarFlow' }}</div>
            <div v-if="message.type === 'ai'" class="markdown" v-html="renderMarkdown(message.content)"></div>
            <div v-else class="plain-text">{{ message.content }}</div>
          </div>
        </div>
        <div v-if="draft || busy" class="message-row ai"><div class="message-avatar">S</div><div class="message-body"><div class="message-name">ScholarFlow <span class="thinking">正在研究</span></div><div v-if="draft" class="markdown" v-html="renderMarkdown(draft)"></div><div v-else class="typing-dots"><span></span><span></span><span></span></div></div></div>
        <details v-if="activities.length" class="activity-details"><summary>研究工具动态 · {{ activities.length }} 条</summary><div v-for="(activity, index) in activities" :key="index">{{ activity }}</div></details>
      </div>
    </div>

    <div class="composer-zone">
      <!-- 审批事件会锁定输入框，用户需先批准或拒绝再继续此线程。 -->
      <div v-if="pending" class="approval-card"><strong>操作需要确认</strong><p>{{ pending.summary || pending.description || pending.message || JSON.stringify(pending) }}</p><div><button class="secondary-button" :disabled="busy" @click="submit('deny')">拒绝</button><button class="primary-button" :disabled="busy" @click="submit('approve')">同意并继续</button></div></div>
      <div v-if="error" class="inline-error" role="alert">{{ error }}</div>
      <div class="composer"><textarea v-model="prompt" :disabled="busy || !!pending" rows="2" placeholder="输入你的研究问题，或请智能体检索论文…" @keydown="handleEnter"></textarea>
        <div class="composer-bottom"><div class="composer-controls"><label>智能体 <select :value="workspace.selectedAgent" :disabled="busy || !!workspace.activeConversation?.title && workspace.activeConversation?.title !== '新研究对话'" @change="workspace.setAgent(($event.target as HTMLSelectElement).value)"><option v-for="agent in workspace.info?.agents || []" :key="agent.key" :value="agent.key">{{ agent.key }}</option></select></label><span class="control-divider"></span><label>模型 <select :value="workspace.selectedModel" :disabled="busy" @change="workspace.setModel(($event.target as HTMLSelectElement).value)"><option v-for="model in workspace.info?.models || []" :key="model" :value="model">{{ model }}</option></select></label></div><button class="send-button" :disabled="busy || !prompt.trim() || !!pending" aria-label="发送问题" @click="submit()">↑</button></div>
      </div>
      <p class="composer-note">回答由模型生成，请通过来源页码核对重要结论。按 Enter 发送，Shift + Enter 换行。</p>
    </div>
  </div>
</template>
