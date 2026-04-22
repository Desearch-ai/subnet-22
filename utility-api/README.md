# SN22 Utility API

Dataset & utility API for [Subnet-22 (Desearch)](https://github.com/desearch-ai) validators.

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

### `GET /dataset/random`

Returns random questions from the dataset.

| Param         | Type        | Default | Description                                   |
| ------------- | ----------- | ------- | --------------------------------------------- |
| `count`       | int (1-256) | 10      | Number of questions                           |
| `search_type` | enum        | —       | Filter: `ai_search`, `x_search`, `web_search` |

```bash
# Get 20 random AI search questions
curl "http://localhost:8000/dataset/random?count=20&search_type=ai_search"
```

## Importing from HuggingFace

Use the import script to load questions from any HuggingFace dataset:

```bash
# Example: Import SQuAD questions for AI search with wikipedia + web tools
python -m app.scripts.import_huggingface \
    --dataset "squad" \
    --split "train" \
    --column "question" \
    --search-types ai_search \
    --ai-tools wikipedia web \
    --limit 5000

# Example: Import general questions for all search types
python -m app.scripts.import_huggingface \
    --dataset "web_questions" \
    --split "train" \
    --column "question" \
    --search-types ai_search web_search x_search

# Example: Short queries for X search only
python -m app.scripts.import_huggingface \
    --dataset "your_dataset" \
    --column "query" \
    --search-types x_search \
    --limit 10000
```

The script deduplicates questions and inserts in batches. Adapt the `--column` flag to match the dataset structure.

## Project Structure

```
app/
├── main.py                  # FastAPI app + lifespan
├── config.py                # Settings from .env
├── db/
│   └── session.py           # Async SQLAlchemy engine & session
├── models/
│   ├── enums.py             # SearchType, AISearchTool enums
│   ├── question.py          # SQLAlchemy Question model
│   └── schemas.py           # Pydantic response schemas
├── routes/
│   └── dataset.py           # /dataset/random endpoint
└── scripts/
    └── import_huggingface.py  # HuggingFace → Postgres import
```

## Database Schema

Single table `questions`:

| Column          | Type                    | Notes                                                                                |
| --------------- | ----------------------- | ------------------------------------------------------------------------------------ |
| id              | UUID                    | Primary key                                                                          |
| query           | text                    | Question text                                                                        |
| search_types    | `search_type_enum[]`    | `ai_search`, `x_search`, `web_search`                                                |
| ai_search_tools | `ai_search_tool_enum[]` | Nullable. `twitter`, `web`, `reddit`, `hacker_news`, `youtube`, `arxiv`, `wikipedia` |
| source          | varchar                 | Origin: `huggingface:squad`, `desearch`, etc.                                        |
| created_at      | timestamp               | Auto-set                                                                             |
