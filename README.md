# 🎓 智能学术研究平台

面向科研文献检索、管理与问答的 Multi-Agent 平台。系统串联论文搜索、PDF 下载、知识库构建和带来源页码的回答，并提供可切换的多知识库。前端采用 Vue 3、TypeScript、Vite、Vue Router 和 Pinia，提供研究问答与论文知识库工作台。

## 项目简介

基于 FastAPI 提供论文搜索、下载、建库、切库和问答接口；LangGraph Supervisor 路由搜索、下载和 RAG Agent，支持从用户研究问题到论文证据的连续工作流。PDF 经解析、清理和切分后，写入 ChromaDB 或 Qdrant，同时建立持久化 BM25 索引。问答时融合向量检索与 BM25 的候选片段，可选用 PageIndex 文档树进一步定位原文页面，再将两类证据交给问答 Agent。

### 技术亮点

- **可追溯的混合检索**：向量检索覆盖语义改写，BM25 补足模型名、公式符号和编号等词项匹配；用 RRF 合并两路排名，并保留论文来源和页码。
- **片段与页面双层证据**：PageIndex 在候选论文内定位页面，系统提取被引用的原文页，与传统 RAG 片段共同进入回答上下文；PageIndex 出错时回退到片段检索。
- **增量与历史知识库兼容**：新文档同时进入向量与词项索引，上传的 PDF 原件保存在知识库内；旧 ChromaDB/Qdrant 库首次使用时可自动补建 BM25 索引，PageIndex 可按需补建。多个知识库独立存放和切换。
- **章节感知论文切分**：清理参考文献和图页碎片，识别章节与小节，按段落组织证据；超长段落按 Token 上限切分。片段保留章节路径、起止页码、来源及图页预览，方便追查回答依据。
- **可复现的检索评测**：按标注页计算 Page Recall、MRR 和 P95 检索耗时；评测脚本可分别比较 BM25、向量、RRF 混合检索和混合检索加 PageIndex。
- **Vue 研究工作台**：对接 FastAPI 的流式问答、会话历史、敏感操作确认、PDF 批量上传和知识库切换；浏览器端按 SSE 事件增量展示回复，并对 Markdown 回答做安全清理。

### 检索效果与目标

已完成的离线冒烟评测使用 **1 篇 15 页的 Transformer 论文和 10 个手工标注问题**。旧版固定长度索引有 34 个片段，BM25 的 Page Recall@5 = 10/10、MRR@5 = 0.883、P95 检索耗时约 1–2 ms。新章节切分生成 28 个片段；在临时 BM25 索引上，Page Recall@5 = 10/10、MRR@5 = 0.900。这只是单篇论文的小样本核验，不能证明总体检索效果提升，也不能推算 PageIndex 增益。

| 指标 | 扩展评测目标（预估，未实测） | 评测口径 |
| --- | --- | --- |
| 混合检索 Page Recall@5 | ≥ 0.82 | 至少 20 篇论文、200 个标注问题；与相同语料的向量单路检索对照 |
| PageIndex 增益 | 跨章节/跨页问题的证据页召回提升 **5–10 个百分点** | 与混合检索对照，固定证据字符/Token 预算；记录额外耗时 |
| 引用页准确率 | ≥ 0.90 | 抽检回答引用的页面是否实际支持对应陈述 |
| 无依据回答率 | ≤ 0.05 | 在不可回答问题集上统计虚构论文事实的比例 |
| 在线检索 P95 | ≤ 15 秒（不含建树和最终生成） | 并发 10，分别报告向量、BM25、PageIndex 和总耗时 |

上述数值是下一阶段的验收目标，不是本仓库当前的实测成绩。复现离线结果：

```sh
PYTHONPATH=backend/app uv run python scripts/eval_retrieval.py data/vector_databases/attention_trial_20260929
```

配置可用的 Embedding/LLM 后，先为知识库建 PageIndex，再加 `--online` 跑向量、混合和 PageIndex 对照。当前配置的百炼账户返回欠费错误，因此在线对照与 PageIndex 提升尚未测得。

## 核心功能

### 1. 文献搜索与下载

- **OpenReview 搜索**：搜索 ICML、NeurIPS、ICLR 等顶级会议的论文
- **arXiv 下载**：支持通过 arXiv ID 或 URL 直接下载论文
- **自动保存**：下载的论文自动保存到 `./data/downloads/papers/` 目录

### 2. 向量数据库管理

- **PDF 转向量库**：将下载的 PDF 论文转换为向量数据库（支持 ChromaDB 和 Qdrant）
- **论文解析**：跳过参考文献并清理图页碎片；识别章节后按段落组合，超长段落按 Token 切分，保存章节路径和起止页码。PDF 默认上限 512 Token、重叠 64 Token（使用 `cl100k_base` 估算，具体模型 Token 数可能不同）；上传和 Agent 建库共用此流程
- **图页预览**：包含图注的页面会保存到数据库目录下的 `previews/`，可在 Streamlit 中查看和下载；图片本身尚未做视觉语义索引
- **多数据库支持**：可以创建和管理多个论文数据库
- **数据库切换**：支持在不同论文数据库之间切换查询

Vue 页面可列出当前服务可见的本地知识库，并显示原始文件数及 BM25 索引状态。上传同名知识库会覆盖旧库，因此前端会在提交前拦截同名名称。知识库切换由后端的服务级配置实现，会影响连接该服务的所有会话。

### 3. 智能问答（RAG）

- **混合检索**：分别使用向量检索和 BM25 关键词检索，通过 RRF 融合排名，兼顾语义改写与模型名、论文编号等精确匹配
- **PageIndex 页面检索**：可选地用文档树定位论文页面，与混合检索得到的片段一并提供给问答 Agent；只传递引用页原文，不把 PageIndex 的生成答案当作证据
- **内容问答**：根据论文内容回答具体问题
- **引用支持**：回答中包含论文引用信息

BM25 索引保存在对应向量库目录下的 `bm25.sqlite3`（Qdrant 集合使用独立文件）。已有向量库首次查询时会根据库内文档自动补建索引；新建库和上传文档时会同时写入两种索引。

章节识别目前基于 PDF 文本层的标题规则，借鉴 PageIndex 的层级结构思路；PageIndex 文档树仍独立用于检索阶段的页面定位，没有直接参与向量片段的切分。扫描版 PDF 需先 OCR。已有向量库不会自动重新切块，需要用新库名重新建库才能得到 `section_path`、`page_start`、`page_end` 元数据。

PDF 切分流程：`逐页提取与正文清理 → 识别章节/小节 → 按段落组织 → 超长段落按 Token 切分 → 保存章节路径、起止页码和来源`。跨页但属于同一章节的短段落可进入同一片段；如果中间页面被过滤，切分器会重置章节路径，避免把附录或图页归入前一节。

启用 PageIndex 本地模式：运行 `uv sync --extra pageindex`，配置有效的 `OPENAI_API_KEY`，并设置 `PAGEINDEX_ENABLED=true`。可分别用 `PAGEINDEX_INDEX_MODEL` 和 `PAGEINDEX_CHAT_MODEL` 指定建树和查树模型。使用 OpenAI 兼容服务时，还可配置 `PAGEINDEX_LLM_API_KEY`、`PAGEINDEX_LLM_BASE_URL` 和对应模型名（例如 `openai/qwen-plus`）；这些变量仅用于 PageIndex 建树和查树，不影响向量 Embedding。启用后，新建 PDF 知识库时会建立 PageIndex 文档树；查询时先用向量与 BM25 找候选论文，再让 PageIndex 定位原文页。PageIndex 本地版依赖 PDF 文本层；图像或扫描页仍需 OCR。已有知识库可运行 `uv run --extra pageindex python scripts/build_pageindex.py <db_path> <pdf路径...>` 补建；Qdrant 库另加 `--collection <集合名>`。

### 4. 一体化工作流

通过 **Paper Research Supervisor** Agent，可以一键完成：
1. 搜索并下载论文
2. 从 PDF 创建向量数据库
3. 基于论文内容回答问题

## 快速开始

### 环境要求

- Python 3.12（项目已通过 `.python-version` 固定版本）
- Node.js 24（Vue 前端本地开发；使用 Docker 时无需本机安装）
- 至少一个 LLM API Key（OpenAI、Groq 等）

### 安装步骤

```sh
# 1. 克隆项目
git clone <repository-url>
cd agent-service-toolkit

# 2. 安装依赖（推荐使用 uv）
curl -LsSf https://astral.sh/uv/install.sh | sh
./scripts/setup.sh

# 或使用 pip
pip install -e .
```

### 配置环境变量

运行安装脚本后会从 `.env.example` 创建 `.env`。默认使用假模型，便于无密钥启动验证；正式使用时至少配置一个模型提供商：

```sh
# 关闭假模型，并配置至少一个 API Key
USE_FAKE_MODEL=false
OPENAI_API_KEY=your_openai_api_key

# 可选：使用本地 embedding 模型（节省 API 费用）
USE_LOCAL_MODEL=True

# 可选：改用阿里百炼 text-embedding-v3 构建和检索向量库
EMBEDDING_PROVIDER=dashscope
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3
DASHSCOPE_API_KEY=your_dashscope_api_key
# 如密钥对应其他地域，在控制台复制 API Host 并配置：
# DASHSCOPE_HTTP_BASE_URL=https://your-api-host/api/v1

# 可选：向量数据库配置
VECTOR_DB_TYPE=chroma  # 或 qdrant
QDRANT_PATH=./data/vector_databases
CHROMA_DB_PATH=./data/vector_databases

# 可选：内容安全检查（需要 Groq API Key）
GROQ_API_KEY=your_groq_api_key
```

切换嵌入模型后，已有向量库需要用同一模型重新建库；不要直接用新模型查询旧索引。

### 启动服务

**方式 1：直接运行**

```sh
# 启动 FastAPI 服务
uv run python backend/app/run_service.py

# 在另一个终端启动 Vue 前端
cd frontend
npm ci
npm run dev
```

**方式 2：使用 Docker**

```sh
docker compose up --build
```

访问：
- Vue 本地开发：http://localhost:5173
- Docker Vue 页面：http://localhost:3000
- API 服务：http://localhost:8080
- API 文档：http://localhost:8080/redoc

Vue 开发服务器通过 Vite 代理 `/api` 和 `/health` 到 `127.0.0.1:8080`；Docker 中由 Nginx 转发到 `agent-service:8080`，因此浏览器与 API 使用同一站点地址。若设置了 `AUTH_SECRET`，可在左侧“接口设置”输入 Bearer Token；Token 仅存于浏览器会话存储。旧 Streamlit 界面仍可通过 `uv run streamlit run backend/app/streamlit_app.py` 或 `docker compose --profile legacy up streamlit` 启动，地址为 `http://localhost:8501`。

### Vue 前端结构

```text
frontend/src/
├── api/          # FastAPI 请求、POST SSE 分片解析和接口类型
├── stores/       # Pinia：服务元数据、知识库及本地对话列表
├── views/        # 研究问答、论文知识库两个页面
├── App.vue       # 导航侧栏和接口设置
└── router.ts     # Vue Router 页面路由
```

对话列表保存在浏览器 `localStorage`，消息历史由 FastAPI 的检查点接口读取；清除浏览器数据会清除本地列表，但不会删除服务端检查点。PDF 上传会在后端解析和建库，耗时取决于论文数量及 Embedding/PageIndex 配置。

## 使用示例

### 完整工作流示例

使用 **Paper Research Supervisor** 完成从搜索到问答的完整流程：

```python
from client import AgentClient

client = AgentClient(agent_id="paper-research-supervisor")

# 一句话完成：搜索、下载、创建数据库、回答问题
response = client.invoke(
    "帮我下载 Transformer 论文（arXiv:1706.03762），"
    "然后根据论文内容回答：Transformer 架构的主要创新是什么？"
)
```

### 分步骤使用

**1. 搜索和下载论文**

```python
from client import AgentClient

client = AgentClient(agent_id="openreview-agent")

# 搜索论文
response = client.invoke("搜索关于 large language model inference optimization 的论文")

# 下载论文
response = client.invoke("下载 arXiv:2309.06180 的论文")
```

**2. 创建向量数据库**

```python
from client import AgentClient

client = AgentClient(agent_id="rag-assistant")

# 从下载的 PDF 创建向量数据库
response = client.invoke(
    "从文件 ./data/downloads/papers/[2309.06180] Efficient Memory Management for Large Language Model Serving with PagedAttention_2309.06180.pdf 创建向量数据库"
)
```

**3. 查询论文内容**

```python
from client import AgentClient

client = AgentClient(agent_id="rag-assistant")

# 基于论文内容回答问题
response = client.invoke("根据论文内容，PagedAttention 是什么？它如何解决内存管理问题？")
```

## 项目结构

```
.
├── backend/app/
│   ├── agents/                    # Agent 定义
│   │   ├── paper_research_supervisor.py  # 监督者 Agent（推荐使用）
│   │   ├── openreview_agent.py           # 论文搜索和下载 Agent
│   │   ├── rag_assistant.py              # RAG 问答 Agent
│   │   ├── tools.py                      # 工具函数（搜索、下载、数据库等）
│   │   └── agents.py                     # Agent 注册
│   ├── core/                     # 核心模块
│   │   ├── llm.py                # LLM 配置
│   │   └── settings.py           # 设置管理
│   ├── service/                  # FastAPI 服务
│   ├── client/                   # 客户端
│   └── streamlit_app.py          # 旧版 Streamlit 界面
├── frontend/                     # Vue 3 研究工作台
│   └── src/                      # 页面、Pinia 状态与 API 客户端
├── data/                         # 数据目录
│   ├── downloads/papers/         # 下载的论文 PDF
│   └── vector_databases/         # 向量数据库存储
└── tests/                        # 测试文件
```

## 可用的 Agents

`auto` 是默认模式。`literature-review` 用于多篇论文综述，会核验检索来源，并在下载论文或建立向量库前暂停等待确认。其他模式仍可选择。

### 文献综述、Skill 与 MCP

仓库内的 `backend/app/agents/skills/` 存放 `SKILL.md`。综述 Agent 启动时读取名称与描述，撰写时按需加载正文。检索结果中的来源链接用于生成引用；来源不足时会提示证据有限。

本地 MCP Server 只提供搜索、已下载论文列表和本地检索，不提供下载或建库：

```sh
PYTHONPATH=backend/app uv run python -m service.mcp_server
```

要让综述 Agent 作为客户端连接本地 stdio Server，在 `.env` 中配置连接及允许使用的工具。例如在项目根目录启动服务时：

```sh
MCP_RESEARCH_SERVERS='{"local":{"transport":"stdio","command":".venv/bin/python","args":["-m","service.mcp_server"],"env":{"PYTHONPATH":"backend/app"}}}'
MCP_RESEARCH_ALLOWED_TOOLS='{"local":["search_arxiv","search_openreview"]}'
```

连接失败不会阻断内置检索，状态可在 `/health` 的 `mcp` 字段查看。外部 MCP 工具只接受部署时配置的搜索工具。审批可在 Streamlit 中点击批准/拒绝，也可在同一 `thread_id` 上调用 Agent API，传入 `approval: "approve"` 或 `"deny"`。

1. **paper-research-supervisor**（推荐）
   - 功能：协调完成完整的文献研究工作流
   - 用途：搜索、下载、创建数据库、回答问题一站式完成

2. **openreview-agent**
   - 功能：专门用于搜索和下载学术论文
   - 用途：从 OpenReview 和 arXiv 搜索并下载论文

3. **rag-assistant**
   - 功能：RAG 问答助手
   - 用途：创建向量数据库、查询论文内容、回答问题

## 技术栈

- **LangGraph**: Agent 框架，实现多 Agent 协调
- **FastAPI**: RESTful API 服务
- **Vue 3 + TypeScript + Vite**: 研究问答与论文知识库前端
- **Vue Router + Pinia**: 页面导航与工作区状态管理
- **Streamlit**: 保留的旧版界面
- **ChromaDB/Qdrant**: 向量数据库，存储论文向量
- **LangChain**: LLM 集成和工具调用
- **LlamaGuard**: 内容安全检查（可选）

## 数据存储

所有数据统一存储在 `./data/` 目录下：

- `./data/downloads/papers/` - 下载的论文 PDF 文件
- `./data/vector_databases/` - 向量数据库文件

建库工具会在当前后端进程中切换到新库。若重启后仍要使用该库，请将 `.env` 中的 `CHROMA_DB_PATH` 或 `QDRANT_PATH` 设置为建库结果返回的 `db_path`。

查看 Chroma 中实际存储的切分文本、metadata、向量维度和图页预览：

```bash
uv run streamlit run scripts/inspect_chroma.py --server.address 127.0.0.1 --server.port 8503
```

## 开发指南

### 本地开发

```sh
# 创建虚拟环境
uv sync --frozen
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 运行服务
uv run python backend/app/run_service.py

# 运行 Vue 前端（另一个终端）
cd frontend && npm ci && npm run dev
```

### 运行测试

```sh
uv run pytest

# 运行特定测试
uv run pytest tests/agents/test_paper_research_supervisor.py

# 前端类型检查、流式事件测试和生产构建
cd frontend && npm test && npm run build
```

## 常见问题

### 如何切换向量数据库？

使用 `rag-assistant` 的 `Get_Vector_DB_Info` 工具查看所有数据库，然后使用 `Switch_Vector_DB` 切换。

### 支持哪些论文来源？

- OpenReview：ICML、NeurIPS、ICLR 等会议论文
- arXiv：所有 arXiv 论文

### 可以使用本地 embedding 模型吗？

可以，设置 `USE_LOCAL_MODEL=True` 即可使用本地模型（如 BGE），无需 OpenAI API。

## License

MIT License
