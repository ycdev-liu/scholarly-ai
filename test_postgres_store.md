# 测试 PostgreSQL Store 是否在使用

## 方法 1: 使用测试端点（推荐）

访问测试端点来验证 PostgreSQL store：

```bash
# 在浏览器中访问
http://localhost:8080/test-store

# 或使用 curl
curl http://localhost:8080/test-store
```

### 预期响应（使用 PostgreSQL）

```json
{
  "store_type": "AsyncPostgresStore",
  "is_postgres": true,
  "is_in_memory": false,
  "test_key": "test_key_xxxxx",
  "namespace": ["test", "store"],
  "write": {
    "success": true,
    "error": null
  },
  "read": {
    "success": true,
    "data": {
      "timestamp": "2025-12-08 ...",
      "test": true,
      "message": "This is a test to verify PostgreSQL store is working"
    },
    "error": null
  },
  "list": {
    "success": true,
    "key_count": 1,
    "error": null
  }
}
```

### 如果使用 InMemoryStore（SQLite）

```json
{
  "store_type": "InMemoryStore",
  "is_postgres": false,
  "is_in_memory": true,
  ...
}
```

## 方法 2: 直接查询 PostgreSQL 数据库

### 连接到 PostgreSQL 容器

```powershell
# 连接到 PostgreSQL
docker compose exec postgres psql -U postgres -d agent_store
```

### 查看 LangGraph store 创建的表

```sql
-- 查看所有表
\dt

-- 查看 store 相关的表（LangGraph 会创建类似 langgraph_store 的表）
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name LIKE '%store%' OR table_name LIKE '%langgraph%';

-- 查看表中的数据
SELECT * FROM langgraph_store LIMIT 10;
```

### 查看连接信息

```sql
-- 查看当前连接
SELECT * FROM pg_stat_activity WHERE application_name LIKE '%store%';
```

## 方法 3: 查看服务日志

启动服务时，查看日志中是否有：

```
Store initialized successfully (type: AsyncPostgresStore)
```

而不是：

```
Store initialized (type: InMemoryStore, no setup required)
```

## 方法 4: 通过健康检查端点

访问 `/health` 端点：

```bash
curl http://localhost:8080/health
```

查看 `database.store` 部分：

```json
{
  "database": {
    "store": {
      "type": "postgres",
      "purpose": "long-term memory (cross-conversation knowledge)",
      "host": "postgres",
      "database": "agent_store",
      "connection": "ok"
    }
  }
}
```

## 方法 5: 测试数据持久化

### 步骤 1: 写入测试数据

```bash
# 调用测试端点写入数据
curl http://localhost:8080/test-store
```

### 步骤 2: 重启服务

```powershell
docker compose restart agent_service
```

### 步骤 3: 再次读取数据

如果使用 PostgreSQL，数据应该仍然存在。如果使用 InMemoryStore，数据会丢失。

### 步骤 4: 验证数据在数据库中

```powershell
# 连接到 PostgreSQL
docker compose exec postgres psql -U postgres -d agent_store

# 查询数据
SELECT * FROM langgraph_store WHERE namespace LIKE '%test%';
```

## 方法 6: 检查环境变量

确认 `.env` 文件中配置了：

```env
STORE_DB_TYPE=postgres
POSTGRES_HOST=postgres
POSTGRES_DB=agent_store
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

## 验证成功的标志

✅ **使用 PostgreSQL Store 的标志：**
- `store_type` 包含 "PostgresStore" 或 "AsyncPostgresStore"
- `is_postgres: true`
- `is_in_memory: false`
- 写入和读取操作成功
- 重启服务后数据仍然存在
- PostgreSQL 数据库中有对应的表和数据

❌ **使用 InMemoryStore 的标志：**
- `store_type` 是 "InMemoryStore"
- `is_postgres: false`
- `is_in_memory: true`
- 重启服务后数据丢失

## 故障排除

### 如果测试端点返回错误

1. **检查 store 是否配置**：
   - 查看日志确认 store 初始化成功
   - 检查环境变量配置

2. **检查 PostgreSQL 连接**：
   ```powershell
   docker compose exec postgres pg_isready -U postgres
   ```

3. **查看详细错误**：
   ```powershell
   docker compose logs agent_service | Select-String -Pattern "store" -Context 5
   ```

### 如果数据没有持久化

1. 确认使用的是 PostgreSQL store（不是 InMemoryStore）
2. 检查 PostgreSQL 数据卷是否正常：
   ```powershell
   docker volume inspect cholarlyai_postgres_data
   ```
3. 验证数据确实写入了数据库（方法 2）

