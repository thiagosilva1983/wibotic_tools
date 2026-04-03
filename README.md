# Wibotic Shared Weekly Production App

A shared, central-database version of the Weekly Production app.

## What this starter does

- Uses **PostgreSQL / Supabase** as the main shared database
- Keeps **all users on the same state** across computers
- Polls SOS every **1–3 minutes**
- Stores shared data centrally:
  - sales orders
  - line items
  - shipped/open/partial status
  - shortage/buildable analysis
  - priority
  - owner
  - notes
  - timestamps
- Streamlit front-end reads from the same shared database for every user

## Project files

- `app.py` — Streamlit UI
- `sync_service.py` — incremental SOS sync loop
- `schema.sql` — PostgreSQL tables / indexes
- `db.py` — database helpers
- `config.py` — environment config
- `sos_adapter.py` — place to plug in your SOS API logic
- `.env.example` — environment variables template
- `requirements.txt` — Python dependencies

## Recommended architecture

SOS -> sync_service.py -> PostgreSQL / Supabase -> Streamlit app -> all users

## Setup

### 1) Create a shared PostgreSQL database

Good option:
- Supabase Postgres

Copy your database URL.

### 2) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Configure env vars

Copy:

```bash
cp .env.example .env
```

Fill in:
- `DATABASE_URL`
- SOS credentials / tokens / endpoints

### 4) Create the database tables

```bash
psql "$DATABASE_URL" -f schema.sql
```

### 5) Start the sync service

```bash
python sync_service.py
```

### 6) Start the Streamlit app

```bash
streamlit run app.py
```

---

## How the sync works

The sync service:

1. Reads `last_sync_at` from the database
2. Calls the SOS adapter for orders created/updated since that timestamp
3. Upserts headers and lines
4. Recomputes analysis only for changed orders
5. Marks missing/closed/shipped states
6. Saves the new `last_sync_at`

### Default timing

- SOS sync every `120` seconds
- UI auto-refresh every `60` seconds

You can change both in `.env`.

---

## Important note about SOS integration

I do not have your live SOS API implementation inside this environment.

So I built the app structure and the shared database flow for you, and I left the SOS-specific fetch logic inside `sos_adapter.py` in a clean format.

That file shows the exact payload shape the rest of the app expects. You can wire it to your existing `sos_requests` code quickly.

---

## What is shared centrally

Everything below is shared for all users:

- priority
- line green / shipped status
- notes
- owner
- order status
- analysis results
- last sync timestamps
- buildable qty
- shortage summary

That means if one user changes priority or notes, everybody sees it after refresh.

---

## Suggested next upgrades

- add login / per-user change history
- add material analysis from your full BOM explode logic
- add location-aware shortages
- add “Analyze SO” detail expander
- add label/packing/scan workflow
- add audit log table

