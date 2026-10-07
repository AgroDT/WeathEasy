# WeathEasy Helm chart

The chart runs the web API with uvicorn and updates prepared weather data through
`weatheasy download` and `weatheasy correction apply`. It requires Kubernetes
1.27 or later. There are no chart dependencies.

## Configure and render

The image tag defaults to `appVersion` from `Chart.yaml`: currently
`ghcr.io/agrodt/weatheasy:0.3.0`, for the upcoming release. Set `image.tag` to
override the default.

Copy [examples/s3.yaml](examples/s3.yaml), then set each source store, S3 endpoint,
schedules, DEM and model artifact addresses. The
resource settings and bounding box in that file are examples, not sizing guidance.

```sh
cp charts/weatheasy/examples/s3.yaml weatheasy-values.yaml
helm lint charts/weatheasy -f weatheasy-values.yaml
helm template weatheasy charts/weatheasy -f weatheasy-values.yaml
```

After reviewing the rendered manifests, install with:

```sh
helm upgrade --install weatheasy charts/weatheasy \
    --namespace weatheasy --create-namespace -f weatheasy-values.yaml
```

Set `sources.SOURCE.store` for each source you want to use. Unconfigured sources
are omitted from `/variables` and OpenAPI; requests to them return
HTTP 404. Source routes are selected when the application is created.
The API requires at least one configured source. CFS2, ERA5 and CMIP6 have
independent Zarr stores, which can live in the same bucket or different buckets.
API and Job Pods access the corresponding published store without a persistent
volume or node binding. Setting a store does not enable automatic updates.

For ERA5 alone, use [examples/era5-only.yaml](examples/era5-only.yaml):

```sh
helm template weatheasy charts/weatheasy -f charts/weatheasy/examples/era5-only.yaml
```

Enable `sources.era5.update.enabled` only if this deployment should download data.
An existing prepared store needs S3 read access; CDS credentials are needed only
for ERA5 downloads.
Local paths belong to local CLI usage. Models and DEM must already be
published before enabling correction; this chart does not train models or prepare
DEM. CMIP6 preparation and EMDNA remain separate, explicitly invoked workflows.

The API is exposed through a ClusterIP Service by default. External publication
is optional (see below). You can access the Service locally with:

```sh
kubectl -n weatheasy port-forward service/weatheasy-weatheasy 8000:8000
```

Startup, readiness and liveness probes use `/startupz`, `/readyz` and `/livez`,
respectively. Each endpoint returns HTTP 200 with `{"status":"ok"}` when the API
responds, without accessing weather storage. The API has no background
initialization, so readiness succeeds as soon as it can serve requests. These
probes do not certify that a source store is ready during an update. Configure
intervals, timeouts and thresholds through `api.startupProbe`,
`api.readinessProbe` and `api.livenessProbe`.

## Publishing the API

The chart supports both Ingress and Gateway API. Both are disabled by default,
and both require `api.enabled: true`. The chart creates routing resources;
controllers, Gateways, DNS records and certificates are managed separately.
Routes forward to the chart's Service using `service.port`. For direct publication
through a cloud load balancer, set `service.type: LoadBalancer` and supply the
provider's settings through `service.annotations`.

### Ingress

Use [examples/ingress.yaml](examples/ingress.yaml) with your controller's
`ingress.className`, hostnames and an existing TLS Secret in the release namespace:

```sh
helm template weatheasy charts/weatheasy \
    -f weatheasy-values.yaml -f charts/weatheasy/examples/ingress.yaml
```

`ingress.annotations` supports controller-specific settings or certificate
manager annotations. Each host can have multiple paths with a `pathType`.
The API serves paths from `/`; selecting a non-root prefix alone does not
rewrite requests or relocate the application's endpoints.

### Gateway API

Use [examples/gateway.yaml](examples/gateway.yaml) to attach an `HTTPRoute` to an
existing Gateway. Set `httpRoute.parentRefs` (name, optional namespace and listener
`sectionName`) and `httpRoute.hostnames`:

```sh
helm template weatheasy charts/weatheasy \
    -f weatheasy-values.yaml -f charts/weatheasy/examples/gateway.yaml
```

Gateway API CRDs and a compatible controller must be installed. The Gateway
listener must allow routes from the release namespace and accept the configured
hostnames. TLS termination and certificates belong to that listener. Configure
route matching through `httpRoute.matches` and metadata through
`httpRoute.annotations`. The chart does not create a Gateway or ReferenceGrant:
the backend Service is in the same namespace as the HTTPRoute.

## Credentials

The chart references existing Secrets and ConfigMaps through `envFrom` and
`extraEnv`; it does not put secret values into Helm values or create Secrets.
Create the referenced Secrets in the release namespace before installing.

The example expects `weatheasy-s3` to supply `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`. Include `AWS_SESSION_TOKEN` for temporary credentials.
For an S3-compatible service set `s3.endpoint`; for AWS leave it empty. Region and
other SDK settings can be supplied with `extraEnv` or the existing Secret.

ERA5 additionally needs CDS credentials. Its `sources.era5.update.envFrom` can reference
a separate Secret supplying `CDSAPI_URL` and `CDSAPI_KEY`; this Secret is passed
only to ERA5 update containers.

For workload identity, configure `serviceAccount.annotations`, or set
`serviceAccount.create: false` and `serviceAccount.name` to an existing account.
Enable `serviceAccount.automountServiceAccountToken` if your credential mechanism
requires the standard Kubernetes token mount. No Kubernetes API permissions are
needed by WeathEasy itself.

## Scheduled updates

Scheduled CFS2 and ERA5 updates are disabled in the default values. For each enabled source,
one CronJob owns downloading and optional correction:

1. With correction disabled, the Job runs `weatheasy download SOURCE`.
2. With correction enabled, downloading runs as an init container. The main
   container then runs `weatheasy correction apply SOURCE` against the same store.
   Failed downloading prevents correction from starting.

Jobs use `restartPolicy: Never`: a failed Pod is retried by the Job controller,
rerunning the sequence and recovering from durable published state. Local GRIB
and NetCDF scratch files are discarded with the Pod. Adjust `backoffLimit`,
resources and scratch limits for your data volume. A large initial load may span
many scheduled intervals; subsequent runs are skipped while it is active.

`concurrencyPolicy: Forbid` prevents scheduled runs of the same CronJob from
interleaving. CFS2 and ERA5 write separate stores and can run independently.
It is not a distributed storage lock: do not point a second release, manually
created Job or an external writer at the same source groups concurrently.
[Kubernetes also does not guarantee exactly-once CronJob execution](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/#job-creation).

Schedules use `timeZone`, defaulting to `Etc/UTC`. To pause scheduled work, set
`sources.SOURCE.update.suspend: true`. This does not stop an already running Job.

`startingDeadlineSeconds` limits how late a scheduled run may start;
`activeDeadlineSeconds` limits total Job duration, including retries and both
containers. Both default to `null`, so long initial loads have no deadline.
Set `successfulJobsHistoryLimit` and `failedJobsHistoryLimit` to control retained
Jobs (defaults: 3 successful and 1 failed).

An initial or manual run can be created from an enabled CronJob's template:

```sh
kubectl -n weatheasy create job --from=cronjob/weatheasy-weatheasy-cfs2 \
    weatheasy-cfs2-initial
```

Pause its schedule and wait for existing Jobs to finish before doing this. Manual
Jobs are not covered by the CronJob's concurrency policy.

## Pod storage and execution

Containers run as UID/GID 1000 with a read-only root filesystem. A writable
`emptyDir` is mounted at `/var/lib/weatheasy` (home and working directory), with a
second one at `/tmp`. `fsGroup` provides access to these volumes. Nothing published
is kept in them, so Pods can be rescheduled on another node.

Tune `workVolume.sizeLimit` and `tmpVolume.sizeLimit`, as well as container
`ephemeral-storage` resource requests/limits when necessary. `nodeSelector`,
`tolerations`, `affinity`, `imagePullSecrets` and Pod annotations are configurable.

## Values reference

`values.schema.json` validates chart settings during Helm linting, rendering and
installation. Workload-specific references are required only when that workload
is enabled. Kubernetes objects such as resources, affinity and environment entries
remain extensible and are validated by Kubernetes.

<!-- VALUES_START -->

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `image.repository` | string | `"ghcr.io/agrodt/weatheasy"` | Application image repository. |
| `image.tag` | string | `""` | Image tag; empty uses Chart.appVersion. |
| `image.pullPolicy` | string | `"IfNotPresent"` | Kubernetes image pull policy. |
| `imagePullSecrets` | array | `[]` | Existing image pull Secret references. |
| `nameOverride` | string | `""` | Override the chart name used in resource names. |
| `fullnameOverride` | string | `""` | Override the complete resource name (truncated to 48 characters). |
| `s3.endpoint` | string | `""` | S3-compatible endpoint; empty uses the SDK default. |
| `envFrom` | array | `[]` | Existing Secret/ConfigMap references for all containers. |
| `extraEnv` | array | `[]` | Additional Kubernetes EnvVar entries for all containers. |
| `serviceAccount.create` | boolean | `true` | Create a ServiceAccount. |
| `serviceAccount.name` | string | `""` | Existing or created account name; empty uses the release resource name. |
| `serviceAccount.annotations` | object | `{}` | Account annotations, including workload identity configuration. |
| `serviceAccount.automountServiceAccountToken` | boolean | `false` | Mount the standard Kubernetes API token. |
| `podSecurityContext` | object | `{"runAsNonRoot":true,"runAsUser":1000,"runAsGroup":1000,"fsGroup":1000}` | Pod security settings. |
| `securityContext` | object | `{"allowPrivilegeEscalation":false,"readOnlyRootFilesystem":true,"capabilities":{"drop":["ALL"]},"seccompProfile":{"type":"RuntimeDefault"}}` | Container security settings. |
| `nodeSelector` | object | `{}` | Node label constraints. |
| `tolerations` | array | `[]` | Kubernetes scheduling tolerations. |
| `affinity` | object | `{}` | Kubernetes affinity settings. |
| `podAnnotations` | object | `{}` | Annotations on API and update Pods. |
| `workVolume.sizeLimit` | string | `"10Gi"` | Size limit of writable home/work directory emptyDir. |
| `tmpVolume.sizeLimit` | string | `"1Gi"` | Size limit of /tmp emptyDir. |
| `api.enabled` | boolean | `true` | Create API Deployment and Service. |
| `api.replicas` | integer | `1` | Number of API replicas. |
| `api.precision` | integer | `6` | Decimal precision of API results. |
| `api.enableCors` | boolean | `false` | Enable application CORS. |
| `api.resources` | object | `{}` | API container resource requests and limits. |
| `api.startupProbe` | object | `{"initialDelaySeconds":0,"periodSeconds":5,"timeoutSeconds":1,"failureThreshold":30,"successThreshold":1}` | Startup probe timings and thresholds; endpoint is /startupz. |
| `api.readinessProbe` | object | `{"initialDelaySeconds":0,"periodSeconds":10,"timeoutSeconds":1,"failureThreshold":3,"successThreshold":1}` | Readiness probe timings and thresholds; endpoint is /readyz. |
| `api.livenessProbe` | object | `{"initialDelaySeconds":0,"periodSeconds":10,"timeoutSeconds":1,"failureThreshold":3,"successThreshold":1}` | Liveness probe timings and thresholds; endpoint is /livez. |
| `service.type` | string | `"ClusterIP"` | ClusterIP, NodePort or LoadBalancer. |
| `service.annotations` | object | `{}` | Service annotations, including cloud load balancer settings. |
| `service.port` | integer | `8000` | Service port; containers listen on port 8000. |
| `sources.cfs2.store` | string | `""` | Independent CFS2 Zarr store URI; empty disables this source. |
| `sources.cfs2.update.enabled` | boolean | `false` | Create the source CronJob. |
| `sources.cfs2.update.schedule` | string | `"0 5 * * *"` | Cron schedule. |
| `sources.cfs2.update.timeZone` | string | `"Etc/UTC"` | Schedule time zone. |
| `sources.cfs2.update.suspend` | boolean | `false` | Pause scheduled runs; does not stop active Jobs. |
| `sources.cfs2.update.backoffLimit` | integer | `2` | Job retry limit. |
| `sources.cfs2.update.startingDeadlineSeconds` | integer / null | `null` | Maximum lateness for starting a scheduled run; null omits the deadline. |
| `sources.cfs2.update.activeDeadlineSeconds` | integer / null | `null` | Maximum total Job duration, including retries; null omits the deadline. |
| `sources.cfs2.update.successfulJobsHistoryLimit` | integer | `3` | Number of successful Jobs to retain. |
| `sources.cfs2.update.failedJobsHistoryLimit` | integer | `1` | Number of failed Jobs to retain. |
| `sources.cfs2.update.resources` | object | `{}` | Download container resource requests and limits. |
| `sources.cfs2.update.envFrom` | array | `[]` | Source-specific Secret/ConfigMap references, appended to global entries. |
| `sources.cfs2.update.extraEnv` | array | `[]` | Source-specific EnvVar entries, appended to global entries. |
| `sources.cfs2.update.correction.enabled` | boolean | `false` | Apply correction after successful download. |
| `sources.cfs2.update.correction.dem` | string | `""` | Published DEM Zarr URI; required when this update and correction are enabled. |
| `sources.cfs2.update.correction.artifacts` | string | `""` | Published model artifact directory URI; required when this update and correction are enabled. |
| `sources.cfs2.update.correction.resources` | object | `{}` | Correction container resource requests and limits. |
| `sources.era5.store` | string | `""` | Independent ERA5 Zarr store URI; empty disables this source. |
| `sources.era5.update.enabled` | boolean | `false` | Create the source CronJob. |
| `sources.era5.update.schedule` | string | `"0 6 10 * *"` | Cron schedule. |
| `sources.era5.update.timeZone` | string | `"Etc/UTC"` | Schedule time zone. |
| `sources.era5.update.suspend` | boolean | `false` | Pause scheduled runs; does not stop active Jobs. |
| `sources.era5.update.backoffLimit` | integer | `2` | Job retry limit. |
| `sources.era5.update.startingDeadlineSeconds` | integer / null | `null` | Maximum lateness for starting a scheduled run; null omits the deadline. |
| `sources.era5.update.activeDeadlineSeconds` | integer / null | `null` | Maximum total Job duration, including retries; null omits the deadline. |
| `sources.era5.update.successfulJobsHistoryLimit` | integer | `3` | Number of successful Jobs to retain. |
| `sources.era5.update.failedJobsHistoryLimit` | integer | `1` | Number of failed Jobs to retain. |
| `sources.era5.update.resources` | object | `{}` | Download container resource requests and limits. |
| `sources.era5.update.envFrom` | array | `[]` | Source-specific Secret/ConfigMap references, appended to global entries. |
| `sources.era5.update.extraEnv` | array | `[]` | Source-specific EnvVar entries, appended to global entries. |
| `sources.era5.update.area` | array | `[]` | Optional ERA5 bounding box [north, west, south, east]. |
| `sources.era5.update.correction.enabled` | boolean | `false` | Apply correction after successful download. |
| `sources.era5.update.correction.dem` | string | `""` | Published DEM Zarr URI; required when this update and correction are enabled. |
| `sources.era5.update.correction.artifacts` | string | `""` | Published model artifact directory URI; required when this update and correction are enabled. |
| `sources.era5.update.correction.resources` | object | `{}` | Correction container resource requests and limits. |
| `sources.cmip6.store` | string | `""` | Independent CMIP6 Zarr store URI; empty disables this source. |
| `ingress.enabled` | boolean | `false` | Create an Ingress; requires API. |
| `ingress.className` | string | `""` | Ingress controller class. |
| `ingress.annotations` | object | `{}` | Ingress controller annotations. |
| `ingress.hosts` | array | `[{"host":"weatheasy.example.org","paths":[{"path":"/","pathType":"Prefix"}]}]` | Host/path rules; each path requires path and pathType. |
| `ingress.tls` | array | `[]` | TLS host lists and existing Secret names. |
| `httpRoute.enabled` | boolean | `false` | Create an HTTPRoute; requires API and installed Gateway API. |
| `httpRoute.annotations` | object | `{}` | HTTPRoute annotations. |
| `httpRoute.parentRefs` | array | `[]` | Existing Gateway references; required when enabled. |
| `httpRoute.hostnames` | array | `[]` | Hostnames accepted by the route; empty leaves matching to the Gateway. |
| `httpRoute.matches` | array | `[{"path":{"type":"PathPrefix","value":"/"}}]` | Gateway API match objects; empty matches all requests. |
| `global` | object | — (commented out) | Shared Helm values; optional and unset by default. |

<!-- VALUES_END -->

## Updating generated files

Edit descriptions, constraints and defaults in `values.schema.json`, then run
from the repository root:

```sh
uv run --frozen scripts/generate_chart.py
```

`values.yaml` and the table above are generated. Properties without a default
appear as commented placeholders; an explicit `default: null` stays active.
Defaults for arrays and free-form objects are specified as complete values.
Nested objects without their own default are assembled from their properties.
References are expanded by `jsonref` before generating either output,
including local pointers and nested `$id` scopes with HTTP(S) identifiers.
URN identifiers are not supported reliably by jsonref; this chart does not use
them. Sibling keywords next to `$ref` are ignored; put defaults and descriptions in the referenced definition.
Cyclic references and references to unavailable resources are rejected.
Defaults are not inferred from
conditions: `if`/`then`/`else` and other constraints remain in the schema for Helm
validation. An explicit object default takes precedence over child defaults.

For CI, check that the generated files are current without changing them:

```sh
uv run --frozen scripts/generate_chart.py --check
```
