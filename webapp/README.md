# Paddle Gallery

A small FastAPI web app that shows the drone RGB tiles in a coverflow-style
"paddle" gallery and lets you pick one.

## Run

```bash
uv sync
uv run uvicorn webapp.app:app --reload
```

Then open http://127.0.0.1:8000

## How it works

- `app.py` — FastAPI backend. Lists images, serves cached JPEG thumbnails
  (the source tiles are large), streams full-resolution images, and stores the
  current selection in memory.
- `templates/index.html` — the page.
- `static/gallery.js` — the paddle/coverflow interaction (click, arrow keys,
  wheel, and touch swipe).
- `static/styles.css` — styling.

## Configuration

By default the gallery reads from
`data/mixeduse-agricultural-fields/rgb-images`. Point it elsewhere with:

```bash
GALLERY_DIR=/path/to/images uv run uvicorn webapp.app:app --reload
```

## Endpoints

| Method | Path              | Description                        |
| ------ | ----------------- | ---------------------------------- |
| GET    | `/`               | Gallery page                       |
| GET    | `/api/images`     | JSON list of images                |
| GET    | `/thumbs/{name}`  | Cached thumbnail                   |
| GET    | `/images/{name}`  | Full-resolution image              |
| GET    | `/api/selection`  | Current selection                  |
| POST   | `/api/selection`  | Set selection (`{"name": "..."}`)  |
