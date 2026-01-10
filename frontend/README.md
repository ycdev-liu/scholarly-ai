# 前端应用

React + TypeScript + Material-UI 前端应用，为智能学术研究平台提供现代化的 Web 界面。

## 技术栈

- **React 18+**: UI 框架
- **TypeScript**: 类型安全
- **Vite**: 构建工具
- **Material-UI (MUI)**: UI 组件库
- **React Router**: 路由管理
- **Zustand**: 状态管理
- **Axios**: HTTP 客户端

## 快速开始

### 安装依赖

```bash
npm install
```

### 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
VITE_API_BASE_URL=http://localhost:8080
VITE_AUTH_SECRET=  # 可选，如果后端启用了认证
```

### 启动开发服务器

```bash
npm run dev
```

应用将在 http://localhost:5173 启动。

### 构建生产版本

```bash
npm run build
```

构建产物将输出到 `dist/` 目录。

## 项目结构

```
src/
├── api/              # API 客户端
│   ├── client.ts     # HTTP 客户端封装
│   └── types.ts      # API 类型定义
├── components/       # React 组件
│   ├── Chat/         # 聊天界面
│   ├── Paper/        # 论文搜索
│   ├── VectorDB/     # 向量数据库管理
│   ├── Layout/       # 布局组件
│   └── Settings/     # 设置页面
├── stores/           # Zustand 状态管理
├── App.tsx           # 主应用组件
└── main.tsx          # 入口文件
```

## 功能模块

### 1. 聊天界面 (`/`)

- 与 AI Agent 进行对话
- 支持流式响应（SSE）
- 选择不同的 Agent 和模型
- 查看对话历史

### 2. 论文搜索 (`/papers`)

- 搜索 OpenReview 和 arXiv 论文
- 下载论文到本地
- 通过 Agent 完成搜索和下载任务

### 3. 向量数据库管理 (`/vector-db`)

- 查看所有向量数据库
- 创建新的数据库（通过 Agent）
- 切换当前使用的数据库

### 4. 设置 (`/settings`)

- 配置 API 服务地址
- 设置认证密钥
- 切换主题（浅色/深色）

## 开发说明

### API 集成

前端通过 `src/api/client.ts` 中的 `ApiClient` 类与后端 FastAPI 服务通信。所有 API 调用都通过这个客户端进行。

### 状态管理

使用 Zustand 进行状态管理，stores 位于 `src/stores/` 目录：

- `chatStore.ts`: 聊天相关状态
- `paperStore.ts`: 论文相关状态
- `vectorDBStore.ts`: 向量数据库状态
- `appStore.ts`: 应用全局状态

### 路由

使用 React Router 进行路由管理，路由配置在 `src/App.tsx` 中。

## 注意事项

- 确保后端 FastAPI 服务运行在配置的地址（默认 http://localhost:8080）
- 某些功能（如文件上传、数据库切换）需要通过 Agent 调用实现
- 前端与 Streamlit 界面可以同时运行，互不干扰
