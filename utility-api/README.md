# SN22 Utility API

Metrics & logging API for [Subnet-22 (Desearch)](https://github.com/desearch-ai) validators.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Postgres and configure .env
cp .env.example .env
# Edit .env with your DB_URL

# 3. Run the API (tables auto-create on startup)
uvicorn app.main:app --reload
```

## API Endpoints

### `POST /logs`

Persist a batch of miner response logs (organic + scoring traffic) submitted by validators.

### `GET /logs/scoring`

Fetch grouped scoring logs by epoch / search type / miner UID.

### `POST /logs/organic/search`

Batch lookup of organic logs by exact request query within a time range.

### `GET /miners` and `GET /miners/{hotkey}`

Aggregate miner state across the configured `VALIDATOR_URLS`.

## Project Structure

```
app/
├── main.py                          # FastAPI app + lifespan
├── config.py                        # Settings from .env
├── auth.py                          # Hotkey-signature auth dep
├── db/session.py                    # Async SQLAlchemy engine & session
└── domains/
    ├── logs/                        # Miner response logging
    │   ├── enums.py                 # QueryKind, SearchType
    │   ├── router.py                # /logs/* endpoints
    │   ├── schemas.py
    │   └── models/miner_response_log.py
    └── miners/                      # Cross-validator miner aggregation
        ├── client.py
        ├── router.py                # /miners/* endpoints
        └── schemas.py
```

## Database Schema

Single table `miner_response_logs` capturing every organic and scoring miner call (request query, search type, validator/miner identity, status, reward payload, response payload). See `app/domains/logs/models/miner_response_log.py` for the full schema and indexes.
