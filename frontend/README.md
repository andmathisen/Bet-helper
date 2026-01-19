### Bet Helper Frontend (MVP)

This is a small React UI that reads from the FastAPI server.

### Prereqs
- Node 18+
- Backend running: `uvicorn bet_helper.api.main:app --host 127.0.0.1 --port 8000`

### Run

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.

### Notes
- Vite proxies `/api/*` to `http://127.0.0.1:8000` (see `vite.config.ts`).
- Use backend commands:
  - `python3 -m bet_helper.cli scrape --league LaLiga`
  - `python3 -m bet_helper.cli predict --league LaLiga`

