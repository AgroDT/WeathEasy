ARG WEATHEASY_TAG=latest

FROM ghcr.io/agrodt/weatheasy:$WEATHEASY_TAG
RUN apt-get update \
    && apt-get install -y --no-install-recommends cron \
    && rm -rf /etc/cron.*/* \
    && rm -rf /var/lib/apt/lists/*
COPY --chmod=644 cron.d /etc/cron.d
ENTRYPOINT [ "cron", "-f" ]
