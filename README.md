# WeathEasy

A service for easy weather data acquiring. It uses [Zarr](https://zarr.readthedocs.io/en/stable/)
as data storage for fast time series queries.

## Installation

We highly recommend to use a virtual environment:

```sh
python3 -m venv /path/to/venv
. /path/to/venv/bin/activate
```

On Windows:

```powershell
python3 -m venv C:\path\to\venv
C:\path\to\venv\Scripts\activate
```

Install WeathEasy from pypi.org:

```sh
python3 -m pip install weatheasy
```

To install the latest (possibly unreleased) version from the GitHub repository:

```sh
python3 -m pip install git+https://github.com/AgroDT/WeathEasy.git
```

Check installation:

```sh
python3 -m weatheasy -v
```

## Extras

The base package supports local queries of prepared weather data, CMIP6
preparation, and DEM preparation. Install extras for the workflows you use:

| Extra | Enables |
| --- | --- |
| `s3` | S3 access for weather data, DEM, model artifacts, station inputs and results |
| `web` | The API launched with `uvicorn weatheasy.web:create_app --factory` |
| `prepare` | CFS2 GRIB decoding and ERA5 downloads through CDS |
| `correction` | Training and applying bias correction models |
| `emdna` | Interpolation of station observations |

For example, querying S3 through the web API needs `s3,web`, while downloading
CFS2 or ERA5 into S3 needs `s3,prepare`:

```sh
python3 -m pip install 'weatheasy[s3,web]'
python3 -m pip install 'weatheasy[s3,prepare]'
```

Reading prepared CFS2 and ERA5 data does not require `prepare`. Applying an
existing model requires `correction`, but does not require `prepare`. Add `s3`
to any workflow that reads or writes an S3 URI. Command help and `--version`
are available with the base package.

## Configuration

WeathEasy web API service is configured with environment variables. Use
[.example.env](./.example.env) as reference. Also you can copy and edit this
file as `.env` or `.local.env`. WeathEasy will load it automatically.

```sh
cp .example.env .env
```

Configure each source independently; for an ERA5-only API, set:

```sh
WEATHEASY__SOURCES__ERA5__STORE=s3://weatheasy/era5.zarr
```

CFS2 and CMIP6 use `WEATHEASY__SOURCES__CFS2__STORE` and
`WEATHEASY__SOURCES__CMIP6__STORE`. Omitted sources require no storage access or
credentials. `/variables` lists configured sources and describes their
variables, and `/era5`, `/cfs2`, `/cmip6` read the corresponding store. A query to
an unconfigured source returns HTTP 404, and its route is absent from OpenAPI
and interactive API documentation. Restart the API after changing the configured
sources. Reading ERA5 requires no CDS credentials.
The API creates stores for configured sources during startup in a worker thread.
Store initialization errors prevent startup. Requests reuse those stores; health
endpoints do not access weather storage.

The API reads prepared stores directly through Zarr's asynchronous API; CLI
queries keep their synchronous implementation. Each API request prepares its
metadata and point indices once, then reads and sends batches of up to 64 days.
The successful response remains a JSON array in chronological order, with
missing values represented as `null`. The first batch is read before sending
HTTP headers; subsequent batches are read as the client consumes the response.

API reads have a cumulative 60-second budget for storage work and waiting for a
read slot across all batches. Formatting and sending data to the client do not
consume that budget. Each Zarr metadata, array read or listing operation
(including backend retries) has a 10-second timeout. Set
`WEATHEASY__STORAGE_READ_TIMEOUT` and `WEATHEASY__STORAGE_OPERATION_TIMEOUT` to
positive finite seconds to override these defaults. Validation errors and
initial read timeouts retain their normal HTTP responses (422 and 504).
After headers are sent, a storage failure terminates the stream: the client
receives a truncated JSON response, and the HTTP status cannot be changed.

The read limiter is held only while preparing metadata or reading a batch and
is released before sending data. Slow clients therefore do not occupy storage
slots. Cancellation propagates to an active asynchronous read; queued cancelled
requests never start I/O. Zarr may internally use threads for filesystem
operations and codecs; cancellation cannot forcibly stop an already running
filesystem call or CPU work. Shared stores stay open when a query finishes,
times out or is cancelled.

CLI applications are configured with command line arguments (see below).
`--store` addresses one source’s store. The default is `./weatheasy/SOURCE.zarr`.
Automatic downloading is configured separately through the Helm chart’s
`sources.SOURCE.update` settings.

## Launching

Before running WeathEasy you need to download weather and climate data. As of
late 2024 the CFSv2 reanalysis and actual forecast require a total about 65GiB
of space. CMIP6 requires about 390GiB of space.

It is also worth updating CFSv2 daily. Just run the download command on a
schedule. WeathEasy will download missing reanalysis data and update the
forecast if necessary. See [below](#download-cli) for details.

‼️ The initial download of data may take a long time: from several days to
several weeks, depending on network bandwidth, the state of the source data
servers, and (possibly) star positions.

| Command | Required extra for local data |
| --- | --- |
| `query`, `list-vars`, `dem`, `download cmip6` | Base package |
| `download cfs2`, `download era5` | `prepare` |
| `correction train`, `correction apply` | `correction` |
| `emdna temperature`, `emdna precipitation` | `emdna` |

All CLI commands can be invoked as `weatheasy ...` or `python -m weatheasy ...`.
The web application is launched separately with `uvicorn weatheasy.web:create_app --factory`.

### Download CLI

To download weather and climate data run:

```sh
weatheasy download {cfs2,cmip6,era5} --store STORE
```

Where `STORE` can be a local path or an S3 link in format
`s3://<bucket>/<prefix>`.

For S3 you need to export `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. For
any not-AWS S3 storages you also need to export `AWS_ENDPOINT_URL_S3`. See the
[Boto3 docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables)
for details.

Optionally you can provide a local path or `s3://<bucket>/<prefix>` to
`--download-cache` to preserve original downloaded files (GRIB2 for CFSv2,
GRIB for ERA5-Land, and NetCDF for CMIP6). Local directories are used directly. With S3,
new files are downloaded locally and uploaded to the cache before processing,
so a processing failure does not require downloading from the source again.
Cached files are fetched only when needed. Remote-cache working copies remain on
local disk across failed runs and are removed after their prepared results commit.
Use `--work-dir` to choose this directory. Defaults are
`$XDG_STATE_HOME/weatheasy/downloads` (otherwise `~/.local/state/weatheasy/downloads`)
on Linux, `%LOCALAPPDATA%\weatheasy\downloads` (otherwise
`~\AppData\Local\weatheasy\downloads`) on Windows, and
`~/Library/Application Support/weatheasy/downloads` on macOS.
Before each run, the loader checks which files are still needed from the committed
results and current request parameters. Only unneeded files older than
`--work-max-age-days` (default: 7) are removed; needed files are retained regardless
of age. Abandoned `.part` downloads are discarded and fetched again. Local
`--download-cache` directories require no second cache and are never cleaned by
this working-copy policy. Working files are grouped by source, remote cache and
prepared-result store; each group contains a `.weatheasy-cache.json` identifying its settings without
credentials. Changing the remote cache or result store creates a separate group;
retired groups can be removed explicitly after abandoning the corresponding job.
Run only one job per source at a time; remote cache keys are assumed to contain
immutable downloads.

For ERA5, use `--bbox NORTH WEST SOUTH EAST` in decimal degrees EPSG:4326
to limit the download region, without antimeridian crossing:

```sh
weatheasy download era5 --store ./era5.zarr --bbox 56 37 55 38
```

ERA5-Land requests use the native GRIB format, avoiding server-side NetCDF
conversion. The loader opens fields lazily through xarray/cfgrib and decodes
one hourly spatial field at a time, reducing inputs into one day of output. A day spans multiple
Zarr chunks, up to `(1, 128, 128)` each; GRIB fields are not repeatedly decoded
for individual spatial tiles. Memory depends on the requested grid, not the
number of days in a month. Source CRS, NaNs and next-midnight accumulations are
preserved. The month is published only after every daily write and raw-cache
upload succeeds; interrupted months are retried without duplicating dates.
Existing NetCDF cache entries are retained but are not reused by the GRIB loader.

The loader keeps up to three outstanding CDS requests by default. Set
`--max-pending-requests N` to change the window (`1` sends requests sequentially).
Ready results are downloaded independently of submission order within each
month, and the window is refilled before raw-cache uploads. Requests
may be submitted for the current and next month; processing and Zarr writes
remain sequential. Temporary GRIB working copies are removed after their month
commits; a persistent local download cache is retained.

The target Zarr's root attribute `era5_requests` maps request hashes to CDS job
IDs, saved immediately after submission. Restarts reuse downloaded files or
resume the saved jobs, including jobs still queued/running. Missing or terminally
failed jobs are replaced; network/authentication errors retain their IDs. The
journal drops only the completed month after its data and raw-cache uploads
commit successfully, preserving jobs submitted for the next month.
Run one writer per target store.

A full example for CFSv2:

```sh
weatheasy download cfs2 --store s3://weatheasy/cfs2.zarr --download-cache s3://weatheasy/downloads
```

To display the full help message, run:

```sh
weatheasy download -h
```

### Query CLI

To query downloaded data run:

```sh
weatheasy query {cfs2,cmip6,era5} --store STORE -o PATH begin end latitude longitude var [var ...]
```

Where:

  - `STORE` is the same as for downloading
  - `PATH` is a path to write results to (by default, results are output to
    console)
  - `begin`, `end` are the date range boundaries in ISO format (yyyy-mm-dd)
  - `latitude`, `longitude` are target coordinates in EPSG:4326 coordinate
    reference system (decimal degrees WGS84)

The command line must ends with a space separated list of target variables to
query. To print a full list of available variables run:

```sh
python3 -m weatheasy list-vars
```

A full example for CFSv2, Moscow:

```sh
weatheasy query cfs2 --store s3://weatheasy/cfs2.zarr \
    2024-01-01 2024-01-31 \
    55.75222 37.61556 \
    TMIN TMAX TMP
```

To display the full help message, run:

```sh
python3 -m weatheasy -h
weatheasy query cfs2 -h
weatheasy query cmip6 -h
```

### Web API

After [configuring](#configuration) launch the web application with:

```sh
uvicorn weatheasy.web:create_app --factory
```

By default it would be listening for incoming requests at
http://127.0.0.1:8000.

Interactive API docs are available at
http://127.0.0.1:8000/docs.

Alternative API docs are available at
http://127.0.0.1:8000/redoc.

Read the [Uvicorn docs](https://www.uvicorn.org/deployment/)
for a full list of supported arguments and options.

### Docker

We provide [Docker images](https://github.com/AgroDT/WeathEasy/pkgs/container/weatheasy)
with all installed dependencies to run WeathEasy. Pull it with:

```sh
docker pull ghcr.io/agrodt/weatheasy
```

The default entry point is the web API. Redefine it to use WeathEasy CLI. For
example, to download CFSv2 data with docker run:

```sh
docker run --rm -it \
    --entrypoint weatheasy \
    ghcr.io/agrodt/weatheasy \
    download cfs2 \
    --store s3://weatheasy/cfs2.zarr
```

For Kubernetes, use the [Helm chart](charts/weatheasy/README.md) to run the API
and scheduled updates with durable data in S3.

## Forecast storage and updates

Each source uses its own Zarr v3 store. Within it, CFS2 has
`reanalysis/{flx,pgb}`, `forecast/{flx,pgb}` and `reanalysis/correction`;
ERA5 has `daily` and `correction`; CMIP6 has one group per variable.
Source metadata such as CFS2’s `updated` and ERA5’s `request_bbox` live at the
store root. Existing combined stores need their source groups moved into
separate store roots; there is no automatic migration or legacy-layout fallback.

Prepared data uses Zarr v3. Before writing, rioxarray adds CF CRS and coordinate
metadata, including the source sphere/ellipsoid parameters and WKT in
`spatial_ref`. GRIB decoding remains with cfgrib/ecCodes: Gaussian latitude
coordinates are preserved without resampling or an invented affine GeoTransform.
Open with `decode_coords="all"` to restore the grid mapping and use `.rio.crs`
after importing rioxarray.

Weather arrays use NaN for both the Zarr fill value and CF `_FillValue`.
Empty chunks are not written during either creation or append; missing chunks
therefore remain NaN when read. CMIP6 NetCDF preparation is still eager;
ERA5-Land GRIB preparation uses bounded daily batches. Dask is not required.

CFS2 forecast grids have permanent group paths
`forecast/flx` and `forecast/pgb`, including when the root store is an
S3 prefix such as `s3://weatheasy/cfs2.zarr`. For example, direct xarray consumers can
open the FLX grid with:

```python
import rioxarray  # Registers the .rio accessor.
import xarray as xr

forecast = xr.open_zarr(
    "s3://weatheasy/cfs2.zarr",
    group="forecast/flx",
    consolidated=False,
    chunks=None,
    decode_coords="all",
)
```

Updates replace the forecast in place. Before writing, the downloader clears
the store root’s `updated` readiness attribute and deletes the existing `forecast`
subtree. It restores `updated` only after both grids finish successfully.
Reanalysis is retained; CMIP6 lives in a separate store.

**CFS2 data can be unavailable during replacement.** If preparation fails,
readiness remains unset until a subsequent download command completes successfully.
There are no forecast generations or reader snapshots: requests and already-open
xarray datasets are not protected against concurrent replacement. Coordinate
external jobs and direct readers to avoid this window; publication is not an
atomic multi-object transaction on S3.

## Terrain for bias correction

Build the static Copernicus DEM once and publish it as a spatial Zarr v3 dataset:
Specify `--bbox NORTH WEST SOUTH EAST` in whole degrees EPSG:4326 and an explicit
output path with `--output` (`-o`); both are required.

```sh
weatheasy dem --bbox 56 37 55 38 -o ./copdem.zarr
# Publish directly to S3 using the same command:
weatheasy dem --bbox 56 37 55 38 -o s3://weatheasy/copdem.zarr
```

Source COG tiles are read directly from their public HTTPS addresses using
Rasterio/GDAL range requests; no local or S3 tile cache is created. The output
contains `latitude`, `longitude`, `elevation` (metres), `slope_we` and `slope_sn`
(m/m), with EPSG:4326 and CF metadata written through rioxarray.

Repeat the same command to resume an interrupted build. Root attributes record
`build_config` (bbox, resolution, source, algorithm version and tile block shape) and `build_status`
(`initializing`, `building`, `computing_slopes`, `complete`). Different parameters
or an existing output without build metadata are rejected; choose a new output
path for a different build. Completed builds are left intact.

The retained uint8 `tile_status` array records one value per 1 degree source tile:
0 pending, 1 done, 2 missing (HTTP 404). Initialization writes only coordinates
and metadata; elevation, slope and status chunks are created on demand. Unwritten
fields read as NaN and unwritten statuses as 0. Tiles are assembled in memory into
blocks of 32 x 32 source tiles, limited by the bbox. The output chunk shape is
computed from the resolution: at 0.1 degrees a full block contains 320 x 320
float32 cells (400 KiB before compression).
Each elevation block is written once, followed by its tile statuses in one matching
status chunk. An interrupted block is read again in full; no partial block progress
or intermediate data group is kept. In an interactive terminal, nested tqdm bars
show completed chunks and tiles in the current chunk, including the current tile
name. Resumed builds count previously completed chunks immediately. Progress is
written to stderr and disabled when stderr is not a terminal. Other HTTP/read failures interrupt the build.
Missing tiles and nodata cells are filled with zero, as ocean/no coverage.
Slopes are computed after all tiles finish and are recomputed if interrupted.
Training and correction reject unfinished builds; older DEM datasets without a
build status remain readable. Use only one writer per output; coordinate readers
so they start after the build completes. This is not a concurrent-writer protocol.

Both training and application read the same prepared DEM. Their `--dem` argument
accepts a local Zarr path or S3 URI:

```sh
weatheasy correction train era5 --store s3://weatheasy/era5.zarr \
    --stations ./stations.parquet --dem s3://weatheasy/copdem.zarr --artifacts s3://weatheasy/models
weatheasy correction apply era5 --store s3://weatheasy/era5.zarr \
    --dem s3://weatheasy/copdem.zarr --artifacts s3://weatheasy/models
```

The model directory accepts a local path or S3 URI. It contains separate joblib,
NumPy and manifest files under `<source>/temp` and `<source>/precip`. Application
reads these files directly through a filesystem; a downloaded local model bundle
is unnecessary. Manifests are published after all files and validate their hashes
before correction starts. Coordinate retraining and application to avoid reading
a model while its files are being replaced. Station Parquet files accept local
paths or S3 URIs.

Library training receives station observations as a `pandas.DataFrame`; reading
the Parquet file belongs to the CLI. Training CLI inputs require `station_id`,
`lat`, `lon`, `date`, and `temperature` (°C) / `precipitation` (mm/day), according
to `--only`; only these columns are loaded. Calendar columns `year`/`month`/`day`
are supported by the library API, but the training CLI requires `date`.
Create a prepared `FileStore` with
`weatheasy.file_storage.get_file_store` for model artifacts and pass it to training
or correction.
The correction CLI shares one synchronous S3 filesystem with an asynchronous
adapter for weather and DEM Zarr access, reusing the same S3 client. Other callers
can supply their filesystem when resolving S3 artifact directories.

Create a prepared DEM `Store` with `weatheasy.storage.get_store` and pass it to
`build_copdem`, training or correction. Direct xarray consumers can open the DEM
with `xr.open_zarr(store, consolidated=False, chunks=None, decode_coords="all")`.

## AquaCrop weather

Read prepared ERA5-Land data and calculate daily reference evapotranspiration:

```python
from datetime import date

import zarr

from weatheasy.aquacrop_adapter import get_weather_data

weather = get_weather_data(
    root=zarr.open_group("./data", mode="r"),
    latitude=55.75,
    longitude=37.62,
    altitude=156,  # metres above sea level
    start=date(2020, 6, 1),
    end=date(2020, 6, 30),  # inclusive
    use_corrected=True,
)
```

The result is a NumPy record array with `tmin`, `tmax` (°C), `pr` and `et0`
(mm/day). The adapter reads existing data and computes ET0 in memory. It requires
no training dependencies. Corrected mean temperature and precipitation are selected
independently when their arrays exist; otherwise the raw variable is used.
Missing values within an existing corrected array remain missing. Set
`use_corrected=False` to select both raw variables.

Bias-correction models include a frozen monthly climatology. They remain valid
through ordinary daily appends to the raw Zarr store. If the raw store is deleted
and recreated, retrain both temperature and precipitation models before applying
correction, even when the dates and coordinates are unchanged.

ET0 uses the daily [FAO-56 equations](https://www.fao.org/4/x0490e/x0490e08.htm),
with the selected `TMP_corr`/`TMP` as its mean temperature instead of the standard
`(Tmin + Tmax)/2`. Tmin/Tmax remain raw for output and radiation calculations.
The stored daily `WIND` is converted from 10 m to 2 m; elevation only determines
clear-sky radiation. The agreed cloud factor is limited to 0.05–1, including a
small denominator when clear-sky radiation is zero. Polar-night results are an
approximation. Precipitation below 1 mm/day becomes zero while NaN is preserved;
nonpositive ET0 becomes NaN, preserving the donor adapter's output policy.

## Development

WeathEasy is written in Python and managed by [uv](https://docs.astral.sh/uv).
After cloning this repository, initialize the development environment with:

```sh
uv sync --all-extras --frozen
```

We also recommend to install and use pre-commit:

```sh
uv tool install pre-commit
uv tool run pre-commit install
```

You can develop WeathEasy with locally running S3-compatible object storage
[MinIO](https://min.io/). Launch it with:

```sh
docker compose -f compose.dev.yml up -d
```

Also you need to export next variables (see [.example.env](./.example.env)):

```sh
export AWS_ENDPOINT_URL_S3=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```


### EMDNA station interpolation

Install `weatheasy[emdna]` and run either processor independently:

```sh
weatheasy emdna temperature stations.parquet -o temperature-results
weatheasy emdna precipitation stations.parquet -o precipitation-results
```

Station Parquet, ERA5 input directory, and output directory accept local paths or
`s3://` URIs. S3 locations share one filesystem client within the command. For example:

```sh
weatheasy emdna temperature s3://weatheasy/stations.parquet \
    -o s3://weatheasy/emdna-temperature --era5 s3://weatheasy/era5-daily
```

Library processors receive station observations as a `pandas.DataFrame`, optional
ERA5 data as an `xarray.Dataset`, and a prepared `FileStore` for the output directory.
Observations accept `date` or the `year`/`month`/`day` columns; inputs remain owned
by the caller. The CLI reads Parquet and processes one ERA5 month at a time.
Remote NetCDF inputs are staged on the job's local temporary disk for reading;
each monthly file is reused across its dates and removed when the month finishes.

Optional positional `BEGIN END` dates (ISO format) limit the period. Without them,
the command processes new observation dates after the latest completed output.
The observations require `station_id`, `lat`, `lon`, `date` (or `year`, `month`,
`day`), and either `temperature` (°C) or `precipitation` (mm/day). Optional elevation and slope columns
retain the donor's name matching. The output grid covers 19–180° E, 41–82° N at
0.1° resolution; observations outside this region are excluded.

`--era5 DIR` accepts the existing daily NetCDF contract: files matching
`*YYYYMM*daily.nc`, coordinate `date`, spatial coordinates `latitude`/`longitude`,
and `t2m_mean` (°C), `tp` (m/day), or `precipitation` (mm/day). These files are a
separate input from the hourly download cache and the application's Zarr store.
Only an exact date is used. Missing dates produce a warning and station-only
results; missing ERA5 cells preserve the station estimate and precipitation
probability. Available cells retain the donor's station/ERA5 blend.

Each completed day's NPZ contains the donor output fields plus each ensemble
member's random-field state and the random-generator state. A local result uses
a temporary file in the output directory and atomic replacement. S3 publishes
the complete object after writing finishes; restart state and results are
published together. Repeating the command skips completed dates and resumes
with their state. A failed write or insufficient station count stops the sequence, so the
unfinished day remains retryable. Old donor NPZ files without restart state
cannot resume a new sequence; use a fresh output directory. Use a single writer
per output directory and keep its grid and ensemble size unchanged.

The climatology is rebuilt from the provided CSV on each invocation; existing
outputs are not recalculated when observations change. Temporal state follows
successive processed dates, including gaps in observation dates. Target elevation
and slopes remain zero pending DEM integration. The donor precipitation
cross-correlation formula is preserved (including its reuse of prior state when
`trange_scrf` is supplied); the CLI does not supply that optional field. Processing
the full grid and ensemble holds arrays in memory; no Dask execution is provided.

### Daily retrieval and precipitation units

A date interval returns one row for **each day**, including both endpoints.
Unavailable days remain missing (`NA` in CSV, `null` in JSON, `NaN` in Python).
Hourly interpolation and weekly, monthly, yearly, or full-period weather
aggregation are not provided in this version.

The public Python getters, CLI CSV, and HTTP responses return precipitation in
**mm/day** for every source: CFS2/ERA5 `PRATE`, CMIP6 `pr`, and `PRATE_corr`.
This changes the numeric units of uncorrected precipitation returned by earlier
versions. Consumers must no longer multiply getter results by 86400 or 21600.
Column names, requested column order, and response shapes are unchanged.

Native Zarr arrays retain their source units: CFS2 `PRATE` is the sum of four
six-hour rates (multiply by 21600 for mm/day); ERA5 `PRATE` and CMIP6 `pr` are
rates (multiply by 86400). Corrected precipitation is already mm/day. Direct
xarray/Zarr consumers must account for this difference from public retrieval.
Other units retain their existing source contracts: CMIP6 temperature is K,
CFS2/ERA5 temperature is °C; CFS2 pressure is Pa, ERA5 pressure is hPa.

### Agroclimate summaries

`AgroClimateCalculator` is a separate Python library interface for summarizing an
already selected daily interval. It reports mean actual evapotranspiration and
potential-minus-actual evaporation (both mm/day), mean soil moisture (%), mean
temperature (°C) and humidity (%), total and maximum daily rainfall (mm), and the
longest confirmed dry spell (days, daily precipitation below 1 mm).

```python
from weatheasy.query import Coords, get_era5_data
from weatheasy.agro_clim import AgroClimateCalculator, format_results

variables = ['LHTFL', 'PEVPR', 'SOILW_0-0.1m', 'TMP', 'RH', 'PRATE']
values = get_era5_data(
    root=root,
    begin=begin,
    end=end,
    coords=Coords(55.0, 37.0),
    variables=variables,
)
data = dict(zip(variables, values, strict=True))
results = AgroClimateCalculator(data).calculate_all_parameters(begin, end)
print(format_results(results))
```

The calculator expects daily precipitation and potential evaporation in mm/day,
latent heat flux in W/m² (positive upward), soil moisture as a fraction, temperature
in °C, and relative humidity in %. ERA5 and CFS2 getter results follow these
contracts. The date arguments describe the interval selected by the caller; they
do not slice arrays without a time index. NaN propagates through numeric
summaries. A missing day interrupts a confirmed dry spell. These summaries do
not change the daily weather retrieval interface.
