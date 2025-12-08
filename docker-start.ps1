# Docker 启动脚本 - PowerShell
# 使用方法: .\docker-start.ps1 [command] [streamlit_app]

param(
    [Parameter(Position=0)]
    [ValidateSet("up", "down", "build", "watch", "logs", "restart", "stop")]
    [string]$Command = "up",
    
    [Parameter(Position=1)]
    [string]$StreamlitApp = "streamlit_app.py"
)

# 检查 .env 文件是否存在
if (-not (Test-Path ".env")) {
    Write-Host "警告: .env 文件不存在，正在从 .env.example 创建..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "已创建 .env 文件，请编辑配置后重新运行" -ForegroundColor Green
        Write-Host "按任意键退出..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit
    } else {
        Write-Host "错误: .env.example 文件不存在" -ForegroundColor Red
        exit 1
    }
}

# 设置 Streamlit 应用环境变量
$env:STREAMLIT_APP = $StreamlitApp

switch ($Command) {
    "up" {
        Write-Host "启动 Docker Compose 服务..." -ForegroundColor Green
        docker compose up -d
    }
    "down" {
        Write-Host "停止 Docker Compose 服务..." -ForegroundColor Yellow
        docker compose down
    }
    "build" {
        Write-Host "构建并启动 Docker Compose 服务..." -ForegroundColor Green
        docker compose up --build -d
    }
    "watch" {
        Write-Host "启动 Docker Compose 服务（热重载模式）..." -ForegroundColor Green
        Write-Host "Streamlit 应用: $StreamlitApp" -ForegroundColor Cyan
        docker compose watch
    }
    "logs" {
        Write-Host "查看 Docker Compose 日志..." -ForegroundColor Green
        docker compose logs -f
    }
    "restart" {
        Write-Host "重启 Docker Compose 服务..." -ForegroundColor Yellow
        docker compose restart
    }
    "stop" {
        Write-Host "停止 Docker Compose 服务..." -ForegroundColor Yellow
        docker compose stop
    }
    default {
        Write-Host "未知命令: $Command" -ForegroundColor Red
        Write-Host "可用命令: up, down, build, watch, logs, restart, stop" -ForegroundColor Yellow
    }
}

