ARG PYVERSION=3.12
ARG VENV_DIR=/opt/venv

FROM python:${PYVERSION}-bookworm AS deps
ARG VENV_DIR
COPY requirements.lock .
RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    sed -i 's/-e file:\.//' requirements.lock \
    && uv venv $VENV_DIR \
    && . $VENV_DIR/bin/activate \
    && uv pip install -r requirements.lock

FROM python:${PYVERSION}-slim-bookworm
ARG VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"
WORKDIR /usr/src/weatheasy
COPY --from=deps $VENV_DIR $VENV_DIR
COPY . .
RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    uv pip install --no-cache --no-deps -e .
ENTRYPOINT ["uvicorn", "weatheasy.web:app", "--host", "0.0.0.0"]
