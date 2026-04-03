CREATE TABLE IF NOT EXISTS sync_state (
    state_key TEXT PRIMARY KEY,
    state_value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sales_orders (
    so_number TEXT PRIMARY KEY,
    customer_name TEXT,
    order_date DATE,
    status TEXT,
    shipped_flag BOOLEAN NOT NULL DEFAULT FALSE,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    priority_rank INTEGER NOT NULL DEFAULT 99,
    priority_label TEXT,
    assigned_to TEXT,
    internal_note TEXT,
    source_last_modified TIMESTAMPTZ,
    source_hash TEXT,
    last_analyzed_at TIMESTAMPTZ,
    buildable_qty INTEGER,
    shortage_count INTEGER,
    main_blocker TEXT,
    shipped_line_color TEXT,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sales_order_lines (
    id BIGSERIAL PRIMARY KEY,
    so_number TEXT NOT NULL REFERENCES sales_orders(so_number) ON DELETE CASCADE,
    line_no INTEGER,
    item_number TEXT,
    description TEXT,
    qty_ordered NUMERIC(18,4),
    qty_shipped NUMERIC(18,4),
    qty_remaining NUMERIC(18,4),
    line_status TEXT,
    line_hash TEXT,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (so_number, line_no)
);

CREATE TABLE IF NOT EXISTS so_analysis (
    so_number TEXT PRIMARY KEY REFERENCES sales_orders(so_number) ON DELETE CASCADE,
    buildable_qty INTEGER,
    shortage_count INTEGER,
    main_blocker TEXT,
    missing_parts_json JSONB,
    summary_json JSONB,
    analysis_hash TEXT,
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    so_number TEXT,
    field_name TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_by TEXT,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sales_orders_status ON sales_orders(status);
CREATE INDEX IF NOT EXISTS idx_sales_orders_active ON sales_orders(active);
CREATE INDEX IF NOT EXISTS idx_sales_orders_priority_rank ON sales_orders(priority_rank);
CREATE INDEX IF NOT EXISTS idx_sales_orders_source_last_modified ON sales_orders(source_last_modified);
CREATE INDEX IF NOT EXISTS idx_sales_order_lines_so_number ON sales_order_lines(so_number);
