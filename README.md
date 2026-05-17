# NAS-HPO-Bench-II-Real API

## Installation

* Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
* Run below:

```bash
git clone https://github.com/yoichii/nashpobench2real-api.git
cd nashpobench2real-api
uv sync
uv run nashpobench2api-download
```

## Benchmarking
```
uv sync --group bench
cd bench_algos
uv run python random_search.py
```
