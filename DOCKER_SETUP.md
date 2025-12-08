# Docker 启动指南 - 混合数据库配置

## 快速开始

### 1. 配置环境变量

复制环境变量示例文件：

```powershell
Copy-Item .env.example .env
```

编辑 `.env` 文件，配置以下关键项：

```env
# LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here

# 数据库配置 - 混合模式
# 短期记忆（对话历史）使用 SQLite
CHECKPOINTER_DB_TYPE=sqlite
SQLITE_DB_PATH=/app/data/checkpoints.db

# 长期记忆（跨对话知识）使用 PostgreSQL
STORE_DB_TYPE=postgres

# PostgreSQL 配置
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=postgres  # Docker 内部使用服务名
POSTGRES_PORT=5432
POSTGRES_DB=agent_store
```

### 2. 启动服务

#### 方式 1: 使用启动脚本（推荐）

```powershell
# 启动服务（后台运行）
.\docker-start.ps1 up

# 启动服务（热重载模式，开发时使用）
.\docker-start.ps1 watch

# 构建并启动
.\docker-start.ps1 build
```

#### 方式 2: 直接使用 Docker Compose

```powershell
# 启动所有服务
docker compose up -d

# 启动并查看日志
docker compose up

# 热重载模式（开发时使用）
docker compose watch

# 重新构建并启动
docker compose up --build -d
```

### 3. 验证服务

启动后，访问以下地址验证：

- **Agent Service API**: http://localhost:8080
- **API 文档 (Swagger)**: http://localhost:8080/docs
- **健康检查**: http://localhost:8080/health
- **Streamlit 应用**: http://localhost:8501

## 数据库配置说明

### 混合数据库架构

本项目支持混合数据库配置：

- **短期记忆 (Checkpointer)**: 使用 SQLite
  - 存储位置: Docker 卷 `sqlite_data` → `/app/data/checkpoints.db`
  - 用途: 存储每个对话线程的历史消息
  - 持久化: ✅ 数据持久化到 Docker 卷

- **长期记忆 (Store)**: 使用 PostgreSQL
  - 存储位置: PostgreSQL 容器
  - 用途: 存储跨对话的共享知识和用户偏好
  - 持久化: ✅ 数据持久化到 Docker 卷 `postgres_data`

### 数据库服务

- **PostgreSQL**: 运行在 `localhost:5432`
  - 用户名: `postgres` (可在 .env 中配置)
  - 密码: `postgres` (可在 .env 中配置)
  - 数据库: `agent_store` (可在 .env 中配置)

- **SQLite**: 存储在 Docker 卷中
  - 容器内路径: `/app/data/checkpoints.db`
  - 数据卷: `sqlite_data`

## 常用命令

### 查看服务状态

```powershell
docker compose ps
```

### 查看日志

```powershell
# 查看所有服务日志
docker compose logs

# 查看特定服务日志
docker compose logs agent_service
docker compose logs postgres

# 实时查看日志
docker compose logs -f agent_service
```

### 停止服务

```powershell
# 停止服务（保留数据）
docker compose stop

# 停止并删除容器（保留数据卷）
docker compose down

# 停止并删除所有数据（包括数据卷）
docker compose down -v
```

### 重启服务

```powershell
# 重启所有服务
docker compose restart

# 重启特定服务
docker compose restart agent_service
```

### 查看数据库

```powershell
# 连接到 PostgreSQL 容器
docker compose exec postgres psql -U postgres -d agent_store

# 查看 SQLite 数据库文件（在容器内）
docker compose exec agent_service ls -lh /app/data/
```

## 数据持久化

### PostgreSQL 数据

PostgreSQL 数据存储在 Docker 卷 `postgres_data` 中，即使删除容器，数据也会保留。

### SQLite 数据

SQLite 数据库文件存储在 Docker 卷 `sqlite_data` 中，映射到容器的 `/app/data/checkpoints.db`。

### 备份数据

```powershell
# 备份 PostgreSQL 数据
docker compose exec postgres pg_dump -U postgres agent_store > backup.sql

# 备份 SQLite 数据（从容器复制）
docker compose cp agent_service:/app/data/checkpoints.db ./backup_checkpoints.db
```

### 恢复数据

```powershell
# 恢复 PostgreSQL 数据
docker compose exec -T postgres psql -U postgres agent_store < backup.sql

# 恢复 SQLite 数据（复制到容器）
docker compose cp ./backup_checkpoints.db agent_service:/app/data/checkpoints.db
```

## 故障排除

### 服务无法启动

1. **检查端口占用**:
```powershell
netstat -ano | findstr :8080
netstat -ano | findstr :5432
```

2. **检查 Docker 是否运行**:
```powershell
docker ps
```

3. **查看详细错误日志**:
```powershell
docker compose logs agent_service
docker compose logs postgres
```

### 数据库连接失败

1. **检查 PostgreSQL 是否健康**:
```powershell
docker compose ps postgres
```

2. **检查环境变量配置**:
确保 `.env` 文件中的 `POSTGRES_HOST=postgres`（Docker 内部使用服务名）

3. **测试数据库连接**:
```powershell
docker compose exec agent_service python -c "import psycopg; print('PostgreSQL available')"
```

### 健康检查失败

如果服务显示 "unhealthy"，等待几秒钟让服务完全启动。如果持续失败：

1. 检查服务日志
2. 检查环境变量配置
3. 重新构建镜像: `docker compose up --build`

## 开发建议

1. **开发时使用 `watch` 模式**: 代码修改会自动重载
2. **生产环境使用 `up -d` 模式**: 更稳定，不会自动重载
3. **定期清理未使用的镜像**: `docker system prune -a`
4. **查看资源使用情况**: `docker stats`

## 环境变量完整列表

参考 `.env.example` 文件获取完整的环境变量配置说明。

