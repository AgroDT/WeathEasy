# Running WeathEasy in container

This example shows how to configure Docker Compose to run the WeathEasy web
service and update CFSv2 data on a schedule using a separate service
with a custom image and cron installed. Please consider using some better
solutions in production, such as [swarm-cronjob](https://crazymax.dev/swarm-cronjob/)
for Docker Swarm and [CronJob](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
for Kubernetes.

See [compose.yml](./compose.yml) file for details.

## Configuration

Copy and edit [.example.env](../../.example.env):

```sh
cp ../../.example.env .env
```

To change the CFSv2 update schedule edit [cron.d/weatheasy](cron.d/weatheasy).
This requires rebuilding the stack images:

```sh
docker compose build
```

## Launching

Load `.env` and start the stack:

```sh
. .env
docker compose up -d
```

WeathEasy API is available at http://127.0.0.1:8000.

API documentation (Swagger) is available at http://127.0.0.1:8000/docs.
