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
    && apt-get install -y -qq --no-install-recommends \
        libeccodes-dev \
        libgdal-dev \
        proj-bin
WORKDIR /usr/src/weatheasy

FROM builder AS dependencies-builder
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --no-install-project --no-dev \
        --no-install-package eccodeslib \
        --no-install-package eckitlib \
        --no-binary-package eccodes \
        --no-binary-package netcdf4 \
        --no-binary-package pyproj \
        --no-binary-package rasterio

FROM builder AS wheel-builder
ARG UV_DYNAMIC_VERSIONING_BYPASS='0.0.0'
COPY . .
RUN UV_DYNAMIC_VERSIONING_BYPASS=$UV_DYNAMIC_VERSIONING_BYPASS uv build --wheel

FROM python:${PYVERSION}-slim-${FLAVOUR}
ARG VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH" \
    ECCODES_PYTHON_USE_FINDLIBS=1
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        "$(apt-cache depends libeccodes-dev | awk '$1 == "Depends:" && $2 ~ /^libeccodes[0-9]/ { print $2 }')" \
        "$(apt-cache depends libgdal-dev | awk '$1 == "Depends:" && $2 ~ /^libgdal[0-9]/ { print $2 }')"
COPY --from=dependencies-builder $VENV_DIR $VENV_DIR
RUN --mount=type=bind,from=builder,source=/usr/local/bin/uv,target=/usr/local/bin/uv \
    --mount=type=bind,from=wheel-builder,source=/usr/src/weatheasy/dist,target=/usr/src/weatheasy/dist \
    uv run pip --python "$VENV_DIR/bin/python3" install --no-deps /usr/src/weatheasy/dist/weatheasy*.whl
RUN groupadd --gid 1000 weatheasy \
    && useradd --uid 1000 --gid weatheasy --home-dir /var/lib/weatheasy --create-home weatheasy
WORKDIR /var/lib/weatheasy
USER 1000:1000
ENTRYPOINT ["uvicorn", "weatheasy.web:app", "--host", "0.0.0.0"]
