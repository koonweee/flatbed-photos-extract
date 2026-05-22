# Web App Plan

Build a Dockerized FastAPI + Jinja/HTMX web app around the extractor. The app is single-machine, family-gated with one shared password, and stores all state on local disk under `data/`.

## Target Shape

- FastAPI app with server-rendered Jinja templates and HTMX partial updates.
- One shared password from `APP_PASSWORD`; login sets a session cookie.
- SQLite metadata database at `DATA_DIR/app.db`.
- Uploaded originals, extracted photos, debug PNGs, and zips stored under `DATA_DIR/jobs/<job_id>/`.
- No users or per-user data partitioning.
- Jobs process one or more uploaded flatbed scans into one batch.
- Global scan parallelism controlled by `MAX_PARALLEL_SCANS`.
- Debug generation controlled by env settings.
- Disk-space guard blocks new submissions when free space under `DATA_DIR` is below `MIN_FREE_DISK_GB`.

## Data Layout

```text
data/
  app.db
  jobs/
    20260521-233000/
      inputs/
      output/
        photos/
        metadata.csv
        debug/
      downloads/
        photos.zip
```

## Milestone 1: Extractor Refactor

Move the current CLI implementation behind an importable library while keeping the CLI behavior intact.

Implementation:
- Create package modules such as:
  - `extractor/core.py`
  - `extractor/batch.py`
  - `cli/main.py`
- Use `python -m cli` as the CLI entrypoint.
- Expose a per-scan API suitable for workers:

```python
process_scan(
    input_path,
    photos_dir,
    debug_dir,
    source_stem,
    write_debug=True,
    debug_panel_width=None,
) -> ScanResult
```

Exit criteria:
- Current CLI usage still works.
- Output structure remains unchanged.
- Both drybox scans still produce 22 photos total.
- `--no-debug` still suppresses debug PNGs.

High-value tests:
- CLI smoke test on `drybox-b-1.png` and `drybox-b-2.png`.
- Unit test for output naming with multiple source scans.
- Unit test that `write_debug=False` creates no debug directory or debug PNGs.
- Regression check that metadata rows equal extracted photo count plus header.

## Milestone 2: App Skeleton, Auth, and Persistence

Add the web app foundation.

Implementation:
- FastAPI app with Jinja templates.
- Shared-password login page.
- Signed session cookie.
- SQLite schema for jobs and files.
- Basic pages:
  - login
  - upload
  - active jobs
  - completed jobs
- Dockerfile and docker-compose with `./data:/app/data`.

Exit criteria:
- App starts in Docker.
- Login gates every app route.
- Authenticated users can see upload, active, and completed pages.
- Database initializes automatically.

High-value tests:
- Unauthenticated request redirects to login.
- Wrong password is rejected.
- Correct password creates a usable session.
- App restarts without losing job metadata.
- Docker container can read/write the mounted `data/` directory.
- Verify with `$agent-browser` that login redirects, invalid credentials, valid login, and protected-page navigation behave correctly in a real browser session.

## Milestone 3: Upload UX and Job Creation

Implement the upload flow and disk-space guard.

Implementation:
- Hero upload button supporting one or many image files.
- Client-side preview of selected files.
- Remove button for unwanted files before submit.
- Optional job title.
- Server-side upload validation.
- Create job row and copy uploaded originals into `data/jobs/<job_id>/inputs/`.
- Disable or reject submit when free disk space under `DATA_DIR` is below `MIN_FREE_DISK_GB`.

Exit criteria:
- User can upload multiple scans and create one queued job.
- Removed files are not uploaded.
- Job title defaults to timestamp when blank.
- Low disk space prevents submission with a clear message.

High-value tests:
- Upload one image.
- Upload multiple images.
- Remove one selected image before submit.
- Blank title creates timestamp-based job.
- Server rejects empty submit.
- Server rejects submit when simulated free space is under threshold.
- Verify with `$agent-browser` that file selection previews render, remove controls update the selection, submit creates a queued job, and validation messages are visible and actionable.

## Milestone 4: Queue and Worker Pool

Process jobs with global scan-level parallelism.

Implementation:
- Background scheduler.
- Global `MAX_PARALLEL_SCANS` semaphore.
- Jobs can run concurrently, but total active scan processing never exceeds the env limit.
- Per-input-file status tracking:
  - pending
  - running
  - completed
  - failed
  - cancelled
- Job-level states:
  - queued
  - running
  - completed
  - failed
  - cancelling
  - cancelled
- Rough ETA based on recent average seconds per scan.

Exit criteria:
- Jobs move from queued to running to completed.
- Multiple jobs can be queued.
- No more than `MAX_PARALLEL_SCANS` scans run at once.
- Active page shows queue order, progress, status, and rough ETA.

High-value tests:
- Queue two jobs and verify chronological active ordering.
- Set `MAX_PARALLEL_SCANS=1` and verify only one scan runs at a time.
- Set `MAX_PARALLEL_SCANS=2` and verify two scans can run concurrently.
- Simulate one scan failure and verify job state/failure messaging.
- ETA is present for running jobs and absent or pending for queued jobs.
- Verify with `$agent-browser` that the active jobs page updates progress, queue order, statuses, and ETA without a manual refresh.

## Milestone 5: Abort, Delete, and Cleanup Semantics

Add user controls for job lifecycle.

Implementation:
- Abort button for queued/running jobs.
- Delete button for completed, failed, and cancelled jobs.
- Running scan cancellation should terminate the active scan worker where feasible.
- Delete removes DB rows and job folder.
- Prevent delete while a job is actively running unless it is aborted first.

Exit criteria:
- Queued job can be cancelled before processing.
- Running job can be marked cancelling and stops remaining scans.
- Completed job can be deleted, including files on disk.
- UI updates without manual refresh.

High-value tests:
- Abort queued job.
- Abort running job with pending scans remaining.
- Delete completed job and verify folder is removed.
- Attempt delete on running job and verify it is blocked or converted to abort flow.
- Restart app after cancellation and verify state is consistent.
- Verify with `$agent-browser` that abort/delete controls are reachable, confirmable where needed, disabled or hidden in invalid states, and reflected in the UI without a manual refresh.

## Milestone 6: Job Detail Views and Downloads

Expose completed results.

Implementation:
- Completed jobs list with row click-through.
- Detail view shows:
  - input gallery
  - combined output gallery
  - debug viewer, if debug exists
  - metadata link/download
  - download-all zip
- Generate `downloads/photos.zip` for each completed job.

Exit criteria:
- Completed job opens a responsive gallery.
- Output gallery combines all extracted photos across input scans.
- Debug viewer shows one composite debug PNG per input scan.
- Download-all zip contains final photos and metadata.

High-value tests:
- Detail view for job with debug enabled.
- Detail view for job with `WRITE_DEBUG=false`.
- Zip download includes all output photos.
- Zip download includes metadata.
- Mobile viewport shows usable gallery layout.
- Verify with `$agent-browser` that completed-job navigation, input/output galleries, debug viewer presence/absence, metadata link, and download-all flow work in desktop and mobile viewports.

## Milestone 7: Polish, Docker, and Operational Checks

Make the app suitable for a small home/VPS deployment.

Implementation:
- Document env vars:
  - `APP_PASSWORD`
  - `DATA_DIR`
  - `MAX_PARALLEL_SCANS`
  - `WRITE_DEBUG`
  - `DEBUG_PANEL_WIDTH`
  - `MIN_FREE_DISK_GB`
- Add health endpoint.
- Add basic structured logging for job events.
- Add responsive styling for upload, job lists, galleries, and detail views.
- Update README with Docker usage.

Exit criteria:
- Fresh clone can run with Docker Compose.
- Health endpoint returns success.
- Logs show job lifecycle events.
- README explains setup, env vars, and where files are stored.

High-value tests:
- Docker Compose smoke test.
- App restart while jobs are queued.
- App restart after completed jobs exist.
- Low-disk guard works in Docker-mounted data dir.
- Manual responsive check on mobile and desktop widths.
- Verify with `$agent-browser` that the main workflows remain usable at mobile and desktop viewport sizes, with no overlapping controls, unreadable text, or blocked primary actions.

## Default Configuration

Recommended defaults:

```env
DATA_DIR=/app/data
MAX_PARALLEL_SCANS=2
WRITE_DEBUG=true
DEBUG_PANEL_WIDTH=
MIN_FREE_DISK_GB=10
```

`MAX_PARALLEL_SCANS` can be increased after checking CPU and memory usage on the target host.
