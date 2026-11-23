-- pipeline/src/radar/db/schema.sql
-- Section 4.2: Supabase / Postgres Schema Definition with Row-Level Security

-- 1. Metadata
create table if not exists series_meta (
  id text primary key,              -- stable slug: 'spx', 'us10y', 'nifty50'
  name text not null,
  source text not null,             -- 'fred' | 'yahoo' | 'nse'
  source_symbol text not null,
  grp text not null,                -- 'equity_index' | 'rates' | 'fx' | 'commodity' | ...
  region text,                      -- 'US', 'IN', 'EU', 'Global'
  unit text not null,               -- 'index' | 'pct' | 'bps' | 'usd_bn'
  frequency text not null,          -- 'D' | 'W' | 'M'
  transform_default text,           -- 'log_return' | 'diff' | 'level'
  start_date date,
  license_note text,
  active boolean default true
);

-- 2. Daily points (long format)
create table if not exists series_points (
  series_id text references series_meta(id),
  d date not null,
  value double precision not null,
  is_filled boolean default false,
  primary key (series_id, d)
);

create index if not exists idx_series_points_date on series_points(d);

-- 3. Observations from rules engine
create table if not exists observations (
  id text primary key,
  as_of date not null,
  type text not null,
  asset text,
  severity text not null,
  headline text not null,
  body text not null,
  metrics jsonb not null,
  rule_id text not null,
  code_version text not null,
  created_at timestamptz default now()
);

create index if not exists idx_observations_as_of on observations(as_of);

-- 4. Pipeline runs and audit history
create table if not exists pipeline_runs (
  id bigserial primary key,
  started_at timestamptz not null,
  finished_at timestamptz,
  status text not null,             -- 'ok' | 'partial' | 'failed'
  code_version text not null,
  params_hash text not null,
  stale_series int default 0,
  notes jsonb
);

-- 5. Data health monitoring
create table if not exists series_health (
  series_id text primary key references series_meta(id),
  last_point date,
  coverage_pct real,
  fill_ratio real,
  status text,                      -- 'ok' | 'stale' | 'failed'
  checked_at timestamptz default now()
);

-- Enable Row Level Security (RLS) on all tables
alter table series_meta enable row level security;
alter table series_points enable row level security;
alter table observations enable row level security;
alter table pipeline_runs enable row level security;
alter table series_health enable row level security;

-- Public read policies
create policy "Allow public select on series_meta" on series_meta for select using (true);
create policy "Allow public select on observations" on observations for select using (true);
create policy "Allow public select on series_health" on series_health for select using (true);
create policy "Allow public select on series_points" on series_points for select using (true);

-- 6. Push Pack Tables (Migration 0003_push_pack.sql)
create table if not exists event_calendar (
  id bigserial primary key,
  event_type text not null,
  event_date date not null,
  announce_utc timestamptz,
  decision text,
  source_url text not null,
  notes text,
  unique (event_type, event_date)
);

create table if not exists cot_positions (
  contract_id text not null,
  report_date date not null,
  released_on date not null,
  category text not null,
  long_pos double precision,
  short_pos double precision,
  spread_pos double precision,
  open_interest double precision,
  primary key (contract_id, report_date, category)
);

create table if not exists curve_points (
  curve_id text not null,
  d date not null,
  maturity_months int not null,
  yield_pct double precision not null,
  primary key (curve_id, d, maturity_months)
);

create table if not exists weekly_levels (
  series_id text not null references series_meta(id),
  week_end date not null,
  value double precision not null,
  primary key (series_id, week_end)
);

alter table event_calendar enable row level security;
alter table cot_positions enable row level security;
alter table curve_points enable row level security;
alter table weekly_levels enable row level security;

create policy "Allow public select on event_calendar" on event_calendar for select using (true);
create policy "Allow public select on cot_positions" on cot_positions for select using (true);
create policy "Allow public select on curve_points" on curve_points for select using (true);
create policy "Allow public select on weekly_levels" on weekly_levels for select using (true);

