# Data Availability

This replication package does not redistribute the raw NASA C-MAPSS files. Download the public C-MAPSS dataset from the NASA Prognostics Center of Excellence or NASA Open Data source according to their terms, then place the files under `data/raw/` in the package root before running data validation or full retraining.

Expected raw files:

```text
train_FD001.txt  test_FD001.txt  RUL_FD001.txt
train_FD002.txt  test_FD002.txt  RUL_FD002.txt
train_FD003.txt  test_FD003.txt  RUL_FD003.txt
train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

Run `python scripts/validate_data.py` before reproducing tables or retraining models.

## NASA Battery Source Data

The manuscript also reports a separately selected, prospectively frozen NASA battery audit. The third-party battery source files are not redistributed in this compact GitHub repository. The pre-outcome protocol, unit inventory, split, acquisition details, hashes, and wording rules are registered at <https://osf.io/xqsv8/>. Post-execution derived evidence is included in the complete public archive at <https://osf.io/u76ze/> (DOI `10.17605/OSF.IO/U76ZE`).

The battery task is an end-of-discharge extrapolation benchmark using capacity and cycle index as direct degradation proxies relative to the declared 1.4-Ah endpoint. It is not represented as early-life field prognostics or a competitive model leaderboard.

## N-CMAPSS DS02 Computational Proxy

The strongly downsampled computational-proxy task is derived from NASA N-CMAPSS DS02-006. The raw 2.45 GB HDF5 file and the derived NPZ cache are not redistributed under the conservative artifact policy. The complete OSF evidence archive retains the official source URL, access date, fixed unit split, sampling factor, window definition, schema, SHA-256, validation code, and deterministic preparation route. This compact GitHub repository retains the implementation and high-level acquisition boundary, but not the complete source manifest or cache. Users download the source from NASA under the source terms and rebuild the cache locally.
