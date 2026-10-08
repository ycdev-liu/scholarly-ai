<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { RouterLink, RouterView, useRouter } from 'vue-router'
import { getToken, setToken } from './api/client'
import { useWorkspaceStore } from './stores/workspace'

const workspace = useWorkspaceStore()
const router = useRouter()
const tokenInput = ref(getToken())
const showSettings = ref(false)
const serviceError = ref('')

onMounted(async () => {
  // 启动时分别加载服务能力和知识库；后者失败不妨碍问答页显示。
  try { await workspace.loadInfo() }
  catch (error) { serviceError.value = error instanceof Error ? error.message : '服务连接失败' }
  try { await workspace.loadKnowledgeBases() }
  catch { /* 知识库页会显示独立的错误状态。 */ }
})

function createConversation() {
  // 侧栏新建会话后直接切回问答页。
  workspace.newConversation()
  router.push('/')
}

function chooseConversation(id: string) {
  workspace.selectConversation(id)
  router.push('/')
}

function saveSettings() {
  // API 客户端负责把密钥存入 sessionStorage；随后重试受保护的列表接口。
  setToken(tokenInput.value)
  showSettings.value = false
  workspace.loadKnowledgeBases().catch(() => undefined)
}
</script>

<template>
  <div class="app-shell">
    <!-- 全局侧栏：导航、最近会话以及后端连接状态。 -->
    <aside class="sidebar">
      <RouterLink class="brand" to="/" aria-label="ScholarFlow 首页">
        <span class="brand-mark">S<span class="brand-mark-dot">.</span></span>
        <span class="brand-copy"><strong>ScholarFlow</strong><small>智能学术研究平台</small></span>
      </RouterLink>

      <button class="new-chat" @click="createConversation"><span class="plus">＋</span> 开启新研究 <span class="new-arrow">↗</span></button>

      <div class="sidebar-section-label">工作空间</div>
      <nav class="main-nav" aria-label="主导航">
        <RouterLink to="/" class="nav-item" active-class="active"><span class="nav-icon">⌕</span> 研究问答</RouterLink>
        <RouterLink to="/library" class="nav-item" active-class="active"><span class="nav-icon">▤</span> 论文知识库</RouterLink>
      </nav>

      <div class="sidebar-section-label recent-label">最近对话 <span>{{ workspace.conversations.length }}</span></div>
      <div class="conversation-list">
        <button v-for="item in workspace.conversations" :key="item.id" class="conversation-item"
          :class="{ selected: item.id === workspace.activeThreadId }" @click="chooseConversation(item.id)">
          <span class="conversation-dot"></span><span class="ellipsis">{{ item.title }}</span>
        </button>
        <div v-if="!workspace.conversations.length" class="empty-recent">你的研究对话会显示在这里。</div>
      </div>

      <div class="sidebar-footer">
        <div class="sidebar-status"><span class="status-dot" :class="{ offline: serviceError }"></span>{{ serviceError ? '服务暂不可用' : '研究空间已就绪' }}</div>
        <button class="settings-link" @click="showSettings = true"><span>⚙</span> 接口设置</button>
      </div>
    </aside>

    <main class="main-area"><RouterView /></main>

    <!-- AUTH_SECRET 只在浏览器会话中使用，不写入源码或构建产物。 -->
    <div v-if="showSettings" class="modal-backdrop" @click.self="showSettings = false">
      <div class="modal-card settings-modal" role="dialog" aria-modal="true" aria-label="接口设置">
        <button class="modal-close" aria-label="关闭" @click="showSettings = false">×</button>
        <div class="eyebrow">连接设置</div><h2>访问密钥</h2>
        <p class="muted">仅当后端启用了 AUTH_SECRET 时填写。密钥只保存在当前浏览器会话中。</p>
        <label class="field-label" for="auth-token">Bearer Token</label>
        <input id="auth-token" v-model="tokenInput" type="password" placeholder="输入服务端配置的 AUTH_SECRET" autocomplete="off" />
        <button class="primary-button full" @click="saveSettings">保存设置</button>
      </div>
    </div>
  </div>
</template>
