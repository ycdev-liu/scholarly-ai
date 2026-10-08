<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { switchKnowledgeBase, uploadPapers } from '../api/client'
import type { KnowledgeBase } from '../api/types'
import { useWorkspaceStore } from '../stores/workspace'

const workspace = useWorkspaceStore()
// 上传配置对应 FastAPI /api/documents/upload 的表单字段。
const files = ref<File[]>([])
const dbName = ref(`papers_${new Date().toISOString().slice(0, 10).replaceAll('-', '')}`)
const dbType = ref('chroma')
const chunkSize = ref(512)
const chunkOverlap = ref(64)
const localEmbedding = ref(true)
const busy = ref(false)
const error = ref('')
const success = ref('')
const dragActive = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

const bases = computed(() => workspace.knowledgeBases?.items || [])
const current = computed(() => workspace.knowledgeBases?.current)

onMounted(() => refresh())

async function refresh() {
  // 列表由后端扫描本地知识库生成，页面只保留本次请求的结果。
  try { await workspace.loadKnowledgeBases() }
  catch (cause) { error.value = cause instanceof Error ? cause.message : '知识库加载失败' }
}

function isCurrent(item: KnowledgeBase): boolean {
  if (!current.value) return false
  // 后端路径可能带有 ./ 前缀，先统一格式再比较当前库。
  const normalize = (path: string) => path.replace(/^\.\//, '').replace(/\/$/, '')
  return current.value.db_type === item.db_type && normalize(current.value.db_path) === normalize(item.db_path)
}

function addFiles(incoming: FileList | File[]) {
  // 拖放与文件选择共用校验；同名文件只保留一份。
  const selected = Array.from(incoming).filter(file => file.name.toLowerCase().endsWith('.pdf'))
  if (selected.length !== incoming.length) error.value = '仅支持 PDF 文件'
  files.value = [...files.value, ...selected].filter((file, index, array) => array.findIndex(other => other.name === file.name) === index)
}

function onDrop(event: DragEvent) {
  dragActive.value = false
  if (event.dataTransfer?.files) addFiles(event.dataTransfer.files)
}

async function upload() {
  error.value = ''
  success.value = ''
  // 后端同名建库会删除旧目录，前端先拦截已发现的同名库。
  if (!files.value.length) { error.value = '请先选择 PDF 文件'; return }
  if (!/^[a-zA-Z0-9_-]+$/.test(dbName.value)) { error.value = '知识库名称只能包含字母、数字、下划线和短横线'; return }
  if (bases.value.some(item => item.name === dbName.value)) { error.value = '同名知识库已存在，上传会覆盖原有数据；请换一个名称'; return }
  if (chunkSize.value < 1 || chunkOverlap.value < 0 || chunkOverlap.value >= chunkSize.value) { error.value = 'Token 重叠必须小于分块上限'; return }
  busy.value = true
  try {
    // 后端负责 PDF 解析、章节切分、Embedding、BM25 和可选的 PageIndex 建树。
    const result = await uploadPapers(files.value, { dbName: dbName.value, dbType: dbType.value, chunkSize: chunkSize.value, chunkOverlap: chunkOverlap.value, localEmbedding: localEmbedding.value })
    if (!result.success) throw new Error(result.errors.join('；') || '入库失败')
    success.value = `已处理 ${result.processed_files.length} 个文件，生成 ${result.total_chunks} 个片段${result.switched ? '，并切换为当前知识库' : ''}。`
    if (result.errors.length) success.value += ` 部分文件有问题：${result.errors.join('；')}`
    if (result.switch_error) success.value += ` 切换失败：${result.switch_error}`
    files.value = []
    dbName.value = `papers_${Date.now()}`
    // 上传成功后重新取列表，展示新库和后端返回的当前库状态。
    await workspace.loadKnowledgeBases()
  } catch (cause) { error.value = cause instanceof Error ? cause.message : '上传失败' }
  finally { busy.value = false }
}

async function activate(item: KnowledgeBase) {
  error.value = ''
  success.value = ''
  busy.value = true
  try {
    // 切换发生在后端进程级配置中，结果可能影响其他浏览器会话。
    const result = await switchKnowledgeBase(item.db_type, item.db_path, item.collection_name)
    if (!result.success) throw new Error(result.error || '切换失败')
    await workspace.loadKnowledgeBases()
    success.value = `已切换到 ${item.name}。后续问答将使用该知识库。`
  } catch (cause) { error.value = cause instanceof Error ? cause.message : '切换失败' }
  finally { busy.value = false }
}
</script>

<template>
  <div class="library-page">
    <header class="topbar"><div class="breadcrumb"><span>工作空间</span><span class="slash">/</span><strong>论文知识库</strong></div><div class="topbar-right"><span class="topbar-pill"><span class="status-dot"></span> 文献管理</span><span class="avatar">研</span></div></header>
    <div class="library-scroll">
      <div class="library-head"><div><div class="eyebrow">YOUR RESEARCH LIBRARY</div><h1>论文知识库<span class="accent-dot">.</span></h1><p>把论文整理为可检索、可追溯的研究证据。</p></div><button class="secondary-button refresh-button" :disabled="busy" @click="refresh">↻ 刷新列表</button></div>
      <div v-if="error" class="inline-error" role="alert">{{ error }}</div><div v-if="success" class="inline-success" role="status">{{ success }}</div>
      <div class="library-grid">
        <!-- 左侧上传区：选择论文和建库参数。 -->
        <section class="panel upload-panel"><div class="panel-header"><div class="panel-icon">＋</div><div><div class="eyebrow">BUILD A LIBRARY</div><h2>导入研究论文</h2></div></div><p class="panel-description">上传 PDF，系统按章节和段落切分，并建立向量与 BM25 索引；启用 PageIndex 后也会建立页面索引。</p>
          <input ref="fileInput" class="visually-hidden" type="file" accept=".pdf,application/pdf" multiple @change="addFiles(($event.target as HTMLInputElement).files || []); ($event.target as HTMLInputElement).value = ''" />
          <div class="dropzone" :class="{ dragging: dragActive }" role="button" tabindex="0" @click="fileInput?.click()" @keydown.enter="fileInput?.click()" @dragenter.prevent="dragActive = true" @dragover.prevent="dragActive = true" @dragleave.prevent="dragActive = false" @drop.prevent="onDrop"><span class="drop-icon">⇧</span><strong>拖放 PDF 到这里，或点击选择文件</strong><small>支持批量上传 · 扫描版 PDF 需先 OCR</small></div>
          <div v-if="files.length" class="selected-files"><div v-for="file in files" :key="file.name" class="selected-file"><span>▤</span><span class="ellipsis">{{ file.name }}</span><small>{{ (file.size / 1024 / 1024).toFixed(1) }} MB</small><button :aria-label="`移除 ${file.name}`" @click="files = files.filter(item => item.name !== file.name)">×</button></div></div>
          <div class="upload-options"><label class="field-label">知识库名称<input v-model.trim="dbName" placeholder="例如 transformer_papers" /></label><div class="option-pair"><label class="field-label">存储类型<select v-model="dbType"><option value="chroma">ChromaDB</option><option value="qdrant">Qdrant</option></select></label><label class="field-label">切块上限（Token）<input v-model.number="chunkSize" type="number" min="1" /></label></div><div class="option-pair"><label class="field-label">重叠（Token）<input v-model.number="chunkOverlap" type="number" min="0" /></label><label class="field-label checkbox-label"><input v-model="localEmbedding" type="checkbox" /> 使用本地 Embedding</label></div></div>
          <button class="primary-button full upload-submit" :disabled="busy || !files.length" @click="upload">{{ busy ? '正在解析与建库…' : `导入 ${files.length || ''} 篇论文` }}</button>
        </section>

        <!-- 右侧列表展示后端检测到的知识库和索引状态。 -->
        <section class="panel bases-panel"><div class="panel-header"><div class="panel-icon amber">▤</div><div><div class="eyebrow">AVAILABLE COLLECTIONS</div><h2>已有知识库 <span class="count-badge">{{ bases.length }}</span></h2></div></div><p class="panel-description">选择一个知识库作为问答的证据来源。切换会影响当前服务的所有会话。</p>
          <div v-if="!bases.length" class="empty-bases"><span>◇</span><strong>还没有可用的知识库</strong><p>上传第一批论文，开始建立自己的研究语料。</p></div>
          <div v-else class="base-list"><div v-for="item in bases" :key="item.db_path" class="base-card" :class="{ current: isCurrent(item) }"><div class="base-card-top"><div class="base-symbol">{{ item.db_type === 'chroma' ? 'C' : 'Q' }}</div><div class="base-card-title"><strong>{{ item.name }}</strong><small>{{ item.db_type === 'chroma' ? 'ChromaDB' : 'Qdrant' }} · {{ item.source_count }} 个原始文件</small></div><span v-if="isCurrent(item)" class="active-badge">使用中</span></div><div class="base-card-bottom"><div><span :class="item.has_bm25 ? 'mini-dot green' : 'mini-dot'"></span>{{ item.has_bm25 ? '向量 + BM25' : '向量索引' }}</div><button v-if="!isCurrent(item)" :disabled="busy" @click="activate(item)">切换使用 ↗</button></div></div></div>
        </section>
      </div>
      <div class="library-footnote">提示：PageIndex 是否建树由后端配置决定；列表中的索引状态只展示 BM25 文件检测结果。</div>
    </div>
  </div>
</template>
