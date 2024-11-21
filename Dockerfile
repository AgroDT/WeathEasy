ARG PYVERSION=3.12
ARG VENV_DIR=/opt/venv

FROM python:${PYVERSION}-bookworm AS builder
ARG VENV_DIR
ENV UV_PROJECT_ENVIRONMENT=$VENV_DIR \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN echo 'deb http://httpredir.debian.org/debian sid main' > /etc/apt/sources.list.d/debian-sid.list \
    && apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends libgdal-dev
WORKDIR /usr/src/weatheasy
COPY pyproject.toml uv.lock ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.5.4,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --no-install-project --no-dev \
        --no-binary-package rasterio \
        --no-binary-package netcdf4
COPY . .
ARG SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0'
RUN --mount=from=ghcr.io/astral-sh/uv:0.5.4,source=/uv,target=/bin/uv \
    --mount=from=ghcr.io/astral-sh/uv:0.5.4,source=/uvx,target=/bin/uvx \
    . "$VENV_DIR/bin/activate" \
    && SETUPTOOLS_SCM_PRETEND_VERSION=$SETUPTOOLS_SCM_PRETEND_VERSION uv pip install --no-cache --no-deps .

FROM python:${PYVERSION}-slim-bookworm
ARG VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"
RUN echo 'deb http://httpredir.debian.org/debian sid main' > /etc/apt/sources.list.d/debian-sid.list \
    && apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends libgdal35 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder $VENV_DIR $VENV_DIR
ENTRYPOINT ["uvicorn", "weatheasy.web:app", "--host", "0.0.0.0"]
