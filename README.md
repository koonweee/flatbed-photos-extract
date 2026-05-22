# flatbed-photos-extract

Extract individual photos from flatbed scans that contain one or more photos on
a dark scanner background.

## Production (Docker)

Create a local environment file:

```bash
cp .env.example .env
```

For any shared or public deployment, set `APP_PASSWORD` in `.env` and paste the
generated `SESSION_SECRET` value into `.env`.

```bash
make prod-build  # first run, or after code/template/static/dependency changes
make prod        # start without rebuilding
```

Open `http://localhost:8000`.

Check health and logs:

```bash
curl http://localhost:8000/health
make prod-logs
```

## Local dev

Run the web app locally without Docker:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
make dev
```

Open `http://localhost:8000`.

## Environment variables (for web)

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_PASSWORD` | blank | Shared login password. Leave blank to disable auth. |
| `SESSION_SECRET` | `APP_PASSWORD` when set | Cookie-signing key that keeps login sessions valid across restarts. Generate with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`. |
| `DATA_DIR` | `/app/data` in Docker, `data` locally | Database, uploads, outputs, debug files, and downloads. |
| `MAX_PARALLEL_SCANS` | `2` | Global scan concurrency limit. |
| `WRITE_DEBUG` | `true` | Write per-scan debug PNGs. |
| `DEBUG_PANEL_WIDTH` | blank | Optional debug PNG width in pixels. |
| `MIN_FREE_DISK_GB` | `10` | Reject uploads below this free-space threshold. |

## CLI

Use this for direct batch extraction from the terminal.

```bash
python -m cli scan-1.png scan-2.png --output-dir output
```

The CLI default output directory is `output/`, which is ignored by git.

## Repo Layout

```text
extractor/   extraction library and bundled model
cli/         CLI argument parsing and batch entrypoint
web/         FastAPI app, Jinja templates, static assets, queue, persistence
tests/       automated tests
plans/       planning notes
```

See [extractor/README.md](extractor/README.md) for extraction pipeline details.
