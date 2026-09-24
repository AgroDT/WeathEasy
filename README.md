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

By default WeathEasy is only available from command line interface with local
storage. The following extras are available to extend its functionality:

- `s3` - enables support for S3-compatible storage for WeathEasy data
- `web` - enables web API endpoint

To install WeathEasy with extras run

```sh
python3 -m pip install weatheasy[s3,web]
```

## Configuration

WeathEasy web API service is configured with environment variables. Use
[.example.env](./.example.env) as reference. Also you can copy and edit this
file as `.env` or `.local.env`. WeathEasy will load it automatically.

```sh
cp .example.env .env
```

CLI applications are configured with command line arguments (see below).

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

### Download CLI

To download weather and climate data run:

```sh
python3 -m weatheasy.download -d STORE {cfs2,cmip6}
```

Where `STORE` can be a local path or an S3 link in format
`s3://<bucket>/<prefix>`.

For S3 you need to export `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. For
any not-AWS S3 storages you also need to export `AWS_ENDPOINT_URL_S3`. See the
[Boto3 docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables)
for details.

Optionally you can provide a local path to `--download-dir` to preserve
original downloaded files (GRIB2 for CFSv2 and NetCDF for CMIP6).

A full example for CFSv2:

```sh
python3 -m weatheasy.download -d s3://weatheasy/zarr --download-dir ./downloads cfs2
```

To display the full help message, run:

```sh
python3 -m weatheasy.download -h
```

### Query CLI

To query downloaded data run:

```sh
python3 -m weatheasy -d STORE -o PATH {cfs2,cmip6} begin end latitude longitude var [var ...]
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
python3 -m weatheasy -d s3://weatheasy/zarr cfs2 \
    2024-01-01 2024-01-31 \
    55.75222 37.61556 \
    TMIN TMAX TMP
```

To display the full help message, run:

```sh
python3 -m weatheasy -h
python3 -m weatheasy cfs2 -h
python3 -m weatheasy cmip6 -h
```

### Web API

After [configuring](#configuration) launch the web application with:

```sh
uvicorn weatheasy.web:app
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
    --entrypoint python \
    ghcr.io/agrodt/weatheasy \
    -m weatheasy.download \
    -d s3://weatheasy/zarr \
    cfs2
```

Find an example for Docker Compose at [examples/docker-compose](examples/docker-compose).

## Forecast storage and updates

Prepared data uses Zarr v3. Before writing, rioxarray adds CF CRS and coordinate
metadata, including the source sphere/ellipsoid parameters and WKT in
`spatial_ref`. GRIB decoding remains with cfgrib/ecCodes: Gaussian latitude
coordinates are preserved without resampling or an invented affine GeoTransform.
Open with `decode_coords="all"` to restore the grid mapping and use `.rio.crs`
after importing rioxarray.

Weather arrays use NaN for both the Zarr fill value and CF `_FillValue`.
Empty chunks are not written during either creation or append; missing chunks
therefore remain NaN when read. NetCDF preparation is still eager; Dask is not
required.

CFS2 forecast grids have permanent group paths
`cfs2/forecast/flx` and `cfs2/forecast/pgb`, including when the root store is an
S3 prefix such as `s3://weatheasy/zarr`. For example, direct xarray consumers can
open the FLX grid with:

```python
import xarray as xr

forecast = xr.open_zarr(
    "s3://weatheasy/zarr",
    group="cfs2/forecast/flx",
    consolidated=False,
    chunks=None,
    decode_coords="all",
)
```

Updates replace the forecast in place. Before writing, the downloader clears
`cfs2`'s `updated` readiness attribute and deletes the existing `cfs2/forecast`
subtree. It restores `updated` only after both grids finish successfully.
Reanalysis and CMIP6 are not deleted by forecast replacement.

**CFS2 data can be unavailable during replacement.** If preparation fails,
readiness remains unset until a subsequent download command completes successfully.
There are no forecast generations or reader snapshots: requests and already-open
xarray datasets are not protected against concurrent replacement. Coordinate
external jobs and direct readers to avoid this window; publication is not an
atomic multi-object transaction on S3.

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
python -m weatheasy.emdna temperature stations.csv -o temperature-results
python -m weatheasy.emdna precipitation stations.csv -o precipitation-results
```

Optional positional `BEGIN END` dates (ISO format) limit the period. Without them,
the command processes new observation dates after the latest completed output.
The CSV requires `station_id`, `lat`, `lon`, `year`, `month`, `day`, and either
`temperature` (°C) or `precipitation` (mm/day). Optional elevation and slope columns
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
member's random-field state and the random-generator state. A temporary local
file is replaced after writing finishes; restart state and results are published
together. Repeating the command skips completed dates and resumes with their
state. A failed write or insufficient station count stops the sequence, so the
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
