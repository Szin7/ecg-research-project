# Deployment and verification

## Local and container startup

Follow the README for a local Python 3.12 environment. `requirements-cpu.lock` is a hashed Linux x86-64 CPU lock; `requirements-cpu.in` and `requirements-tested.txt` record its direct inputs. It is not a GPU or universal operating-system lock.

For a Docker-capable host:

```bash
docker compose up --build
```

Open http://localhost:8501 . Stop with `docker compose down`.

The container installs the hashed lock, runs as a non-root user, and has a Streamlit health check. Compose binds to loopback, drops Linux capabilities, disables privilege escalation, and sets CPU/memory limits.

**Docker build and startup have not been executed in this environment because no Docker engine is available.** The CI file includes a container build/health job, but that remote workflow has not been run here. This is deployment configuration, not evidence of a cloud deployment.

## Verification scope

See `artifacts/release_validation.json` for the checks actually completed in this release. Tests cover preprocessing, subject overlap, causal RR features, fixed-context attacks, cache consistency, attribution, and one-to-one peak matching. Browser checks exercise the compact distribution without raw/full research data.

CPU timings report warm-up, batch sizes, thread count, median and p95. They exclude preprocessing, network, cold startup, and rendering. Process peak memory includes imports and data and is not incremental model memory.

The demo has no authentication, patient store, upload service, or clinical monitoring workflow. Public deployment requires a separate operational design with access control, capacity measurement, monitoring, and data-handling decisions.

## Reproducibility

Each training run records its seed, configuration, dataset hashes, code hashes, runtime versions, history, duration, and checkpoint checksum. New runs refuse existing directories. Study files retain selection rules, predictions, error tables, bootstrap intervals, attack configuration, and checksums.

Source hashes describe the files present at each run's start. Later UI, documentation, CLI, and formatting changes can make release hashes differ from earlier training snapshots. Identical seeds do not guarantee bitwise identity across machines or numerical-library versions. The study's frozen evaluation configuration describes the reported attacks.
