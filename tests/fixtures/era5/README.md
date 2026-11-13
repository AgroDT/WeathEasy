Synthetic GRIB2 fixtures created locally with ecCodes from
`regular_ll_sfc_grib2`, centre `ecmf`. They contain no downloaded ERA5 data.
Keep these binaries fixed: tests decode them instead of generating GRIB files.

Both files contain the 24 input variables listed independently in
`test_era5.py::REQUIRED_INPUTS`. The grid is 2 × 3 cells, latitude 55.3, 55.2;
longitude 37.1, 37.2, 37.3; Earth radius 6,367,470 m.

- `day.grib`: 25 hourly valid times from 2000-01-01 00:00 through
  2000-01-02 00:00. The first `t2m` cell has a bitmap missing value.
- `midnight.grib`: the last valid time extracted from `day.grib`, retaining
  the original messages. Accumulated fields have forecast step 24 hours and
  a reference time on the previous day.

Values are constant over space except for the missing cell. `t2m` and `d2m`
are 293.15 K; `sp` is 100,000 Pa. Wind alternates between (3, 4) and (-3, -4)
m/s each hour, separating mean wind speed (5 m/s) from speed of mean wind (0).
Soil/skin temperatures are 280 K and soil moisture is 0.2.

Accumulated precipitation is 0.008 m, deliberately changed to 0.001 m at
23:00 and 0.012 m at the following 00:00. Daily precipitation must use the
following midnight (12 mm), not 23:00. Other values:

| Variable | Value |
| --- | ---: |
| `pev` | -0.004 m |
| `slhf` | -8,640,000 J/m² |
| `sshf` | -4,320,000 J/m² |
| `ssrd` | 8,640,000 J/m² |
| `ssr` | 6,480,000 J/m² |
| `strd` | 17,280,000 J/m² |
| `str` | -864,000 J/m² |
| `sde` | 0.1 m |
| `sd` | 0.02 m |

Downloader/cache tests fake the CDS transport and GRIB-reader boundary with
small NetCDF responses. Actual GRIB decoding, scalar midnight time, NaN/CRS,
invalid responses and bounded hourly reads are covered by these static files.
