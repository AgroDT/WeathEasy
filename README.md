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

## Development

WeathEasy is written in Python and managed by [Rye](https://rye.astral.sh/).
After cloning this repository, initialize the development environment with:

```sh
rye sync --all-features
```

We also recommend to install and use pre-commit:

```sh
rye tools install pre-commit
pre-commit install
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
