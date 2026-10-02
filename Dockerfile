ARG PYVERSION=3.12
ARG FLAVOUR=trixie
ARG VENV_DIR=/opt/venv

FROM python:${PYVERSION}-${FLAVOUR} AS builder
ARG VENV_DIR
ENV UV_PROJECT_ENVIRONMENT=$VENV_DIR \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends libgdal-dev
WORKDIR /usr/src/weatheasy
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --no-install-project --no-dev \
        --no-binary-package rasterio \
        --no-binary-package netcdf4
COPY . .
ARG UV_DYNAMIC_VERSIONING_BYPASS='0.0.0'
RUN . "$VENV_DIR/bin/activate" \
    && UV_DYNAMIC_VERSIONING_BYPASS=$UV_DYNAMIC_VERSIONING_BYPASS uv pip install --no-cache --no-deps .

FROM python:${PYVERSION}-slim-${FLAVOUR}
ARG VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends "$(apt-cache depends libgdal-dev | awk '$1 == "Depends:" && $2 ~ /^libgdal[0-9]/ { print $2 }')"
COPY --from=builder $VENV_DIR $VENV_DIR
ENTRYPOINT ["uvicorn", "weatheasy.web:app", "--host", "0.0.0.0"]
