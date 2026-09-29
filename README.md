# 🎓 智能学术研究平台

一个基于 AI Agent 的智能学术研究系统，帮助学生和研究人员快速查找、下载学术论文，并基于论文内容进行智能问答。

## 项目简介

本项目是一个专门为学术研究设计的智能助手系统，通过 AI Agent 技术实现：

- **文献搜索**：从 OpenReview（ICML、NeurIPS、ICLR 等顶级会议）和 arXiv 搜索学术论文
- **文献下载**：自动下载论文 PDF 文件并保存到本地
- **向量数据库**：将下载的论文转换为向量数据库，支持快速检索
- **智能问答**：基于论文内容进行 RAG（检索增强生成）问答，回答论文相关问题

## 核心功能

### 1. 文献搜索与下载

- **OpenReview 搜索**：搜索 ICML、NeurIPS、ICLR 等顶级会议的论文
- **arXiv 下载**：支持通过 arXiv ID 或 URL 直接下载论文
- **自动保存**：下载的论文自动保存到 `./data/downloads/papers/` 目录

### 2. 向量数据库管理

- **PDF 转向量库**：将下载的 PDF 论文转换为向量数据库（支持 ChromaDB 和 Qdrant）
- **论文解析**：建库时跳过参考文献，清理图页中的重复碎片文字，保留图注和来源页码
- **图页预览**：包含图注的页面会保存到数据库目录下的 `previews/`，可在 Streamlit 中查看和下载；图片本身尚未做视觉语义索引
- **多数据库支持**：可以创建和管理多个论文数据库
- **数据库切换**：支持在不同论文数据库之间切换查询

### 3. 智能问答（RAG）

- **语义搜索**：基于向量数据库进行语义搜索，找到最相关的论文内容
- **内容问答**：根据论文内容回答具体问题
- **引用支持**：回答中包含论文引用信息

### 4. 一体化工作流

通过 **Paper Research Supervisor** Agent，可以一键完成：
1. 搜索并下载论文
2. 从 PDF 创建向量数据库
3. 基于论文内容回答问题

## 快速开始

### 环境要求

- Python 3.12（项目已通过 `.python-version` 固定版本）
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

# 在另一个终端启动 Streamlit Web 界面
uv run streamlit run backend/app/streamlit_app.py
```

**方式 2：使用 Docker**

```sh
docker compose up --build
```

访问：
- Web 界面：http://localhost:8501
- API 服务：http://localhost:8080
- API 文档：http://localhost:8080/redoc

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
│   └── streamlit_app.py          # Web 界面
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
- **Streamlit**: Web 用户界面
- **ChromaDB/Qdrant**: 向量数据库，存储论文向量
- **LangChain**: LLM 集成和工具调用
- **LlamaGuard**: 内容安全检查（可选）

## 数据存储

所有数据统一存储在 `./data/` 目录下：

- `./data/downloads/papers/` - 下载的论文 PDF 文件
- `./data/vector_databases/` - 向量数据库文件

建库工具会在当前后端进程中切换到新库。若重启后仍要使用该库，请将 `.env` 中的 `CHROMA_DB_PATH` 或 `QDRANT_PATH` 设置为建库结果返回的 `db_path`。

## 开发指南

### 本地开发

```sh
# 创建虚拟环境
uv sync --frozen
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 运行服务
uv run python backend/app/run_service.py

# 运行 Web 界面
uv run streamlit run backend/app/streamlit_app.py
```

### 运行测试

```sh
uv run pytest

# 运行特定测试
uv run pytest tests/agents/test_paper_research_supervisor.py
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
