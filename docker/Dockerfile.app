FROM python:3.12.3-slim AS builder

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装 uv
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

# 只安装客户端依赖
RUN uv sync --frozen --only-group client && \
    rm -rf /root/.cache/pip /root/.cache/uv

# 最终阶段
FROM python:3.12.3-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/client/ ./client/
COPY src/schema/ ./schema/
COPY src/streamlit_app.py .
COPY src/arg_app.py .

RUN echo '#!/bin/sh\nstreamlit run "${STREAMLIT_APP:-streamlit_app.py}"' > /app/start.sh && \
    chmod +x /app/start.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

CMD ["/app/start.sh"]
