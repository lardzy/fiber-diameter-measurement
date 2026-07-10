# Area inference service

The container is fail-closed by default. Production deployments must provide a
Bearer token and mount the trusted model directory read-only:

```bash
docker run --rm -p 9001:9001 \
  -e AREA_API_TOKEN='replace-with-at-least-16-random-characters' \
  -v /path/to/area-models:/opt/area_weights:ro \
  area-infer:latest
```

Only local development may opt into anonymous access explicitly:

```bash
docker run --rm -p 9001:9001 \
  -e AREA_ALLOW_ANONYMOUS_DEV=1 \
  -v /path/to/area-models:/opt/area_weights:ro \
  area-infer:latest
```

- `GET /live` checks only the web process.
- `GET /ready` verifies auth configuration, runtime availability, required
  weights, and trusted SHA-256 values.
- `GET /health`, `POST /v1/warmup`, and `POST /v1/infer` require
  `Authorization: Bearer <token>` unless anonymous development mode is enabled.

Default limits are a 64 MiB HTTP request, 48 MiB decoded image bytes, 50 million
decoded pixels, a 1.5 GiB `pixels × candidates` mask working-set budget, one active
inference, and a two-model LRU cache. Authentication is checked before request
bodies are buffered. `AREA_REQUIRED_MODELS` can contain a comma-separated
trusted subset; otherwise readiness scans trusted assets and requires at least
one verified, loadable model.

The image runs as UID/GID `10001`. Bind-mounted weight files must be readable by
that identity (for example, read-only mode `0444` or an equivalent ACL); the
service never needs write permission to the model directory.
