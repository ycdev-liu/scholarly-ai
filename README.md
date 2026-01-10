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

- Python 3.12+
- Node.js 18+ (用于前端开发)
- 至少一个 LLM API Key（OpenAI、Groq 等）

### 安装步骤

```sh
# 1. 克隆项目
git clone <repository-url>
cd agent-service-toolkit

# 2. 安装依赖（推荐使用 uv）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen

# 或使用 pip
pip install -e .
```

### 配置环境变量

创建 `.env` 文件：

```sh
# 必需的 API Key（至少一个）
OPENAI_API_KEY=your_openai_api_key

# 可选：使用本地 embedding 模型（节省 API 费用）
USE_LOCAL_MODEL=True

# 可选：向量数据库配置
VECTOR_DB_TYPE=qdrant  # 或 chroma
QDRANT_PATH=./data/vector_databases
CHROMA_DB_PATH=./data/vector_databases

# 可选：内容安全检查（需要 Groq API Key）
GROQ_API_KEY=your_groq_api_key
```

### 启动服务

**方式 1：直接运行**

```sh
# 启动 FastAPI 服务
python src/run_service.py

# 在另一个终端启动 Streamlit Web 界面（可选）
streamlit run src/streamlit_app.py

# 在另一个终端启动 React 前端（可选）
cd frontend
npm install
npm run dev
```

**方式 2：使用 Docker**

```sh
docker compose watch
```

访问：
- React 前端：http://localhost:5173
- Streamlit 界面：http://localhost:8501
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
├── frontend/                     # React 前端应用
│   ├── src/
│   │   ├── api/                 # API 客户端
│   │   ├── components/          # React 组件
│   │   │   ├── Chat/           # 聊天界面
│   │   │   ├── Paper/          # 论文搜索
│   │   │   ├── VectorDB/       # 向量数据库管理
│   │   │   └── Layout/         # 布局组件
│   │   ├── stores/             # Zustand 状态管理
│   │   └── App.tsx             # 主应用
│   └── package.json
├── src/
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
│   └── streamlit_app.py          # Streamlit Web 界面
├── data/                         # 数据目录
│   ├── downloads/papers/         # 下载的论文 PDF
│   └── vector_databases/         # 向量数据库存储
└── tests/                        # 测试文件
```

## 可用的 Agents

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

### 后端
- **LangGraph**: Agent 框架，实现多 Agent 协调
- **FastAPI**: RESTful API 服务
- **ChromaDB/Qdrant**: 向量数据库，存储论文向量
- **LangChain**: LLM 集成和工具调用
- **LlamaGuard**: 内容安全检查（可选）

### 前端
- **React 18+**: UI 框架
- **TypeScript**: 类型安全
- **Vite**: 构建工具
- **Material-UI (MUI)**: UI 组件库
- **React Router**: 路由管理
- **Zustand**: 状态管理
- **Axios**: HTTP 客户端

### 其他界面
- **Streamlit**: 传统 Web 用户界面（与 React 前端共存）

## 数据存储

所有数据统一存储在 `./data/` 目录下：

- `./data/downloads/papers/` - 下载的论文 PDF 文件
- `./data/vector_databases/` - 向量数据库文件

## 开发指南

### 本地开发

#### 后端开发

```sh
# 创建虚拟环境
uv sync --frozen
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 运行 FastAPI 服务
python src/run_service.py
```

#### 前端开发

```sh
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 配置环境变量（可选）
cp .env.example .env
# 编辑 .env 文件，设置 VITE_API_BASE_URL

# 启动开发服务器
npm run dev
```

前端开发服务器将在 http://localhost:5173 启动。

#### Streamlit 界面（可选）

```sh
# 运行 Streamlit Web 界面
streamlit run src/streamlit_app.py
```

Streamlit 界面将在 http://localhost:8501 启动。

### 前端构建

```sh
cd frontend
npm run build
```

构建产物将输出到 `frontend/dist/` 目录，可以部署到静态文件服务器或集成到 FastAPI 服务中。

### 运行测试

```sh
pytest

# 运行特定测试
pytest tests/agents/test_paper_research_supervisor.py
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
