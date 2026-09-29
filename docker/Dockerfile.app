FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv

# Install only the dependencies needed for the client application
# --frozen: Use exact versions from the lock file
# --only-group client: Only install dependencies marked as part of the "client" group in pyproject.toml
RUN uv sync --frozen --only-group client

COPY backend/app/client/ ./client/
COPY backend/app/schema/ ./schema/
COPY backend/app/streamlit_app.py .
COPY backend/app/arg_app.py .

# Create a startup script that uses environment variable
RUN echo '#!/bin/sh\nstreamlit run "${STREAMLIT_APP:-streamlit_app.py}"' > /app/start.sh && \
    chmod +x /app/start.sh

CMD ["/app/start.sh"]
