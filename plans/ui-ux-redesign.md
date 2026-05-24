# UI/UX Redesign Plan

## Goal

Improve the Flatbed Photos web UI around upload, job history, result review, and image viewing. The redesign should follow the generated contact sheet direction: denser screens, simpler copy, clearer state-specific actions, shared gallery/lightbox behavior, and automatic light/dark mode.

## Design Reference

- Contact sheet: `/home/jtkw/.codex/generated_images/019e4dcb-792d-7310-bd1a-12dad1eee81f/`
- Current desktop screenshots: `tmp/ui-audit/desktop/`
- Current mobile screenshots: `tmp/ui-audit/mobile/`
- The contact sheet is the visual direction for density, layout, theme, and interaction patterns.
- This plan and user comments are the binding requirements when they are more specific than the contact sheet.
- Verification should compare the implemented UI against both the current-state screenshots and the contact sheet, with the contact sheet used as the target design direction.

## Design Principles

- Keep the app tool-first, not marketing-like.
- Reduce vertical spacing and avoid nested panels.
- Prefer one obvious interaction per task.
- Make upload and gallery interactions consistent across upload, active jobs, completed jobs, and job details.
- Keep filenames readable but bounded with truncation and full-name access through `title` text or equivalent accessible labels.
- Use state-specific job history columns instead of one generic table for every status.
- Hide authentication UI when auth is disabled.
- Support automatic dark mode through `prefers-color-scheme`.

## Shared Components

Implementation should identify reusable template, CSS, and JavaScript components before changing individual pages. The goal is to keep interactions uniform and avoid one-off upload, table, row, and gallery behavior.

### App Shell

- Shared topbar, brand, primary nav, optional auth actions, page width, page padding, density, theme variables, and focus styles.
- Hides `Log out` when auth is disabled.
- Allows upload-only page drag state without affecting non-upload pages.

### Form Field

- Shared label/input/hint/error pattern.
- Used by the upload job title input now and future compact forms.
- Makes label ownership visually obvious.
- Supports compact density and dark mode.

### File Dropzone

- Shared upload affordance for click-to-upload and local drag target behavior.
- Text: `Drag images here or click to upload`.
- No visible file type list.
- Keeps native file input accessible.

### Page Drop Overlay

- Shared whole-page drag overlay, enabled on upload page only.
- Dims the page while files are dragged over the viewport.
- Shows short helper text and routes dropped files through the same validation path as file input selection.

### Toast Stack

- Shared transient feedback for upload validation and future async actions.
- Error variant required first for invalid dropped/selected files.
- Toasts should be concise, dismissible or auto-expiring, layout-stable, and themed.

### Upload File List

- Shared dense selected-file list for the upload page.
- Fixed max height with scrolling for large batches.
- Rows include thumbnail, truncated filename, optional size/status text, and compact `X` remove action.
- Clicking a thumbnail or row opens the shared lightbox.

### Thumbnail Grid/List

- Shared image collection pattern for upload files, active job inputs, completed inputs, completed outputs, and debug images.
- Supports dense grid and compact list variants.
- Uses consistent captions, truncation, hover/focus states, and click-to-open behavior.

### Shared Lightbox

- Shared image viewer for every image collection.
- Used by uploaded images, active job inputs, completed inputs, completed outputs, and debug images.
- Owns close `X`, previous/next, filename, count, open/download action, keyboard navigation, and hover-to-zoom.
- Must work on mobile without horizontal overflow.

### Job Collection

- Shared wrapper for active and completed job lists.
- Provides responsive behavior, empty states, and density.
- Supports state-specific row/card content rather than forcing identical columns.
- Can be a shared table/card partial with mode-specific column definitions.

### Job Row/Card

- Shared responsive unit for a single job.
- Desktop can render as a dense row/table row; mobile should render as a compact card-like block.
- Owns title truncation, status chip, primary metadata, and row-level actions.
- Active mode shows progress, ETA, `View inputs`, and abort when allowed.
- Completed mode shows completed time, result availability/counts, detail/download/delete actions.
- Failed/cancelled rows show reason and suppress irrelevant actions.

### Status Chip

- Shared job/file status indicator.
- Accessible contrast in light and dark mode.
- Compact enough not to inflate rows or wrap awkwardly.

### Progress Indicator

- Shared compact progress bar and numeric label for active jobs.
- Should not appear on completed rows unless there is a clear reason.

### Tabs

- Shared tab component for completed detail: `Output`, `Input`, `Debug`.
- Supports focus states and compact mobile layout.
- Keeps inactive content hidden without breaking gallery/lightbox behavior.

### Action Button Group

- Shared action layout for job rows/cards and detail headers.
- Handles primary, secondary, and destructive actions consistently.
- Prevents action columns from squishing content.
- Hides unavailable actions instead of showing disabled clutter unless disabled state communicates useful pending work.

### Filename/Text Truncation Utility

- Shared CSS utility for long names.
- Used by upload rows, job rows, file status rows, gallery captions, and lightbox header/footer.
- Preserves full values through `title` attributes or accessible labels.
- Prevents long filenames from widening tables, cards, or modals.

## Milestone 1: Upload Page Simplification

### Scope

- Use the shared `Form Field`, `File Dropzone`, `Page Drop Overlay`, `Toast Stack`, `Upload File List`, `Thumbnail Grid/List`, `Shared Lightbox`, and `Filename/Text Truncation Utility` patterns.
- Make the whole upload page a drag target.
- When files are dragged over the page, dim the page and show a centered overlay/helper message.
- Replace the current split upload affordance with one unified drop zone:
  - Text: `Drag images here or click to upload`
  - No visible file type list.
- Keep the native file input visually hidden but accessible through the drop zone.
- Change job title placeholder to `Enter job title here (optional)`.
- Ensure the `Job title` label visually belongs to the input below it.
- Add client-side validation for dropped/selected files.
- Show clear toast messages for invalid dropped files.
- Use compact uploaded-file rows with thumbnail, truncated filename, optional size/status text, and `X` remove action.
- Make the uploaded-file list scrollable with a fixed max height suitable for large batches.

### Exit Criteria

- Dragging files anywhere over the upload page shows an overlay.
- Dropping valid images adds them to the uploaded list.
- Dropping invalid files does not add them and shows a toast.
- Selecting files via click still works.
- Uploaded list remains bounded and scrollable with many files.
- Remove buttons are compact `X` buttons.
- Long filenames do not stretch the layout.
- Upload form still submits the selected file list correctly.

### Agent Browser Verification

- Desktop viewport, open `/upload`.
- Capture empty upload page.
- Drag or programmatically upload several valid images.
- Capture selected-file list.
- Simulate invalid file drop or file selection and verify toast text appears.
- Capture overlay during drag if feasible.
- Mobile viewport, repeat empty and selected-file screenshots.
- Compare captures against the contact sheet upload frames for density, label/input relationship, unified dropzone, scrollable upload list, and remove `X` affordance.

## Milestone 2: Shared Gallery And Lightbox

### Scope

- Use the shared `Thumbnail Grid/List`, `Shared Lightbox`, and `Filename/Text Truncation Utility` patterns.
- Extract the current completed-detail gallery/lightbox behavior into shared markup/CSS/JS usable by uploaded images, active job input scans, completed job input scans, completed output photos, and debug images.
- Add hover-to-zoom behavior in the lightbox for pointer devices.
- Use consistent lightbox controls: close `X`, previous/next, image count, filename, and open/download action when applicable.
- Ensure the lightbox works on mobile without horizontal overflow.
- Preserve keyboard navigation: Escape closes, left/right arrows navigate.

### Exit Criteria

- Clicking any uploaded image opens the shared lightbox.
- Clicking active job input gallery opens the same lightbox.
- Completed output/input/debug galleries use the same lightbox.
- Hover zoom works on desktop and does not break mobile.
- Long filenames truncate in the lightbox header/footer without hiding controls.
- Keyboard navigation works.

### Agent Browser Verification

- Desktop `/upload`: select images, click thumbnail, capture lightbox.
- Desktop `/active`: click `View inputs`, click an input thumbnail, capture lightbox.
- Desktop `/jobs/{completed_id}`: open output/input/debug tabs and capture lightbox from each content type.
- Mobile `/upload`, `/active`, and completed detail: open lightbox and capture.
- Compare lightbox captures against the contact sheet shared-lightbox frame for visual hierarchy, close/prev/next placement, filename/count treatment, and hover-zoom affordance.

## Milestone 3: Job History Redesign

### Scope

- Use the shared `Job Collection`, `Job Row/Card`, `Status Chip`, `Progress Indicator`, `Action Button Group`, `Thumbnail Grid/List`, `Shared Lightbox`, and `Filename/Text Truncation Utility` patterns.
- Replace the current one-size-fits-all job table with state-aware views.
- Active jobs should prioritize job title, status, progress, ETA, useful created/started time, `View inputs`, and abort action when allowed.
- Completed jobs should prioritize job title, status, completed time, input/output counts if available, download all when available, delete action, and open/detail action.
- Failed/cancelled completed-history rows should show failure/cancel reason clearly and avoid irrelevant download actions.
- Prevent squished columns on desktop.
- Use compact card-like rows on mobile rather than forcing table columns.
- Truncate long job titles and filenames where needed, preserving full text in accessible labels/tooltips.

### Exit Criteria

- Active and completed job pages use relevant state-specific columns/actions.
- Long titles and many file entries do not distort layout.
- Active jobs expose input gallery access with the same interaction pattern as completed jobs.
- Completed rows do not show active-only ETA/progress fields unless useful.
- Failed/cancelled rows do not show unavailable downloads.
- Mobile layout is readable without horizontal scrolling.

### Agent Browser Verification

- Seed or use jobs in queued, running, completed, failed, and cancelled states.
- Desktop `/active`: capture state-specific active job list.
- Desktop `/completed`: capture completed, failed, and cancelled rows.
- Mobile `/active` and `/completed`: capture compact layouts.
- Verify `View inputs` opens a gallery/lightbox from active jobs.
- Compare active/completed captures against the contact sheet job-history frames for column relevance, truncation, row density, and non-squished responsive behavior.

## Milestone 4: Completed Detail Tabs

### Scope

- Use the shared `Tabs`, `Thumbnail Grid/List`, `Shared Lightbox`, `Action Button Group`, and `Filename/Text Truncation Utility` patterns.
- Redesign completed task detail to use tabs: `Output`, `Input`, and `Debug`.
- Default to `Output`.
- Hide or disable tabs with no content only if that remains clear and predictable.
- Remove Metadata CSV download from the result/detail UI.
- Keep `Download all` only where output zip exists and status is completed.
- Use the shared gallery/lightbox for all tab image grids.

### Exit Criteria

- Completed detail no longer stacks Output, Input, and Debug vertically.
- Output/Input/Debug tabs switch content without page reload where practical.
- Metadata CSV download is not shown in the UI.
- Download all remains available for completed jobs with a zip.
- Empty tab states are compact and clear.

### Agent Browser Verification

- Desktop completed detail: capture each tab.
- Desktop lightbox from Output, Input, and Debug tabs.
- Mobile completed detail: capture tabs and one tab switch.
- Confirm Metadata CSV link is absent.
- Compare detail captures against the contact sheet completed-detail frames for tab layout, action placement, gallery density, and mobile tab usability.

## Milestone 5: Global Density, Auth, And Dark Mode

### Scope

- Use the shared `App Shell`, `Status Chip`, `Action Button Group`, and theme variables across all pages.
- Tighten global spacing: smaller panel padding, tighter topbar, denser tables/cards, compact buttons and status chips.
- Add automatic dark mode using `@media (prefers-color-scheme: dark)`.
- Keep status colors accessible in both themes.
- Remove `Log out` when auth is disabled.
- Preserve current login/logout behavior when auth is enabled.

### Exit Criteria

- Light mode matches the simpler contact sheet direction.
- Dark mode is automatically applied when user preference is dark.
- Auth-disabled sessions do not show `Log out`.
- Auth-enabled sessions still show login/logout correctly.
- Buttons, chips, borders, and focus states remain accessible in both themes.

### Agent Browser Verification

- Desktop light mode screenshots: upload, active, completed, detail.
- Desktop dark mode screenshots using `agent-browser set media dark`.
- Mobile light and dark screenshots for upload and job detail.
- Auth disabled: verify no `Log out`.
- Auth enabled in test mode or local config: verify login/logout still appears and works.
- Compare light and dark captures against the contact sheet palette chips and themed frames for color balance, contrast, spacing, and status-chip treatment.

## Milestone 6: Regression Tests

### Scope

- Update or add tests around rendered UI behavior that is server-side visible:
  - auth-disabled hides logout
  - auth-enabled shows logout
  - completed detail excludes Metadata CSV link
  - active/completed pages render expected state-specific actions
- Keep browser-only behavior covered by agent-browser verification rather than brittle unit tests unless the existing test style supports it.

### Exit Criteria

- Existing test suite passes.
- New tests cover server-rendered behavior changes.
- Browser verification screenshots are saved under `tmp/ui-audit/after/`.

### Commands

```bash
make test
```

### Agent Browser Verification Deliverables

- `tmp/ui-audit/after/desktop-upload.png`
- `tmp/ui-audit/after/desktop-upload-selected.png`
- `tmp/ui-audit/after/desktop-upload-invalid-toast.png`
- `tmp/ui-audit/after/desktop-active.png`
- `tmp/ui-audit/after/desktop-active-input-lightbox.png`
- `tmp/ui-audit/after/desktop-completed.png`
- `tmp/ui-audit/after/desktop-detail-output.png`
- `tmp/ui-audit/after/desktop-detail-input.png`
- `tmp/ui-audit/after/desktop-detail-debug.png`
- `tmp/ui-audit/after/desktop-lightbox-hover-zoom.png`
- `tmp/ui-audit/after/mobile-upload.png`
- `tmp/ui-audit/after/mobile-active.png`
- `tmp/ui-audit/after/mobile-detail-tabs.png`
- `tmp/ui-audit/after/mobile-lightbox.png`
- `tmp/ui-audit/after/dark-upload.png`
- `tmp/ui-audit/after/dark-detail.png`

Each deliverable should be reviewed against the contact sheet path in the Design Reference section. Any intentional divergence should be noted in the implementation summary with the reason.

## Implementation Notes

- Start by creating or consolidating shared components before page-specific polish.
- Prefer shared template partials for gallery/lightbox, job rows/cards, tabs, and action groups if that keeps duplication low.
- Keep JavaScript plain and local to the app; no new frontend build tooling.
- If upload validation logic grows, keep file-type checks in one small helper shared by drop and input-change handlers.
- Use CSS variables for theme colors before adding dark mode.
- Avoid introducing external icon libraries unless the current no-build setup can support them cleanly. Text `X`, `‹`, and `›` controls are acceptable if accessible labels are correct.
- Preserve no-JavaScript basics where feasible: file input and form submission should still work.

