-- pipeline/src/radar/db/migrations/0003_push_pack.sql
-- Migration 0003: Push Pack Schema Extensions
-- Governed by PUSH_PACK_PLAN.md Section 11

-- 1. Central bank event calendar
create table if not exists event_calendar (
  id bigserial primary key,
  event_type text not null,         -- 'fomc' | 'rbi_mpc' | 'ecb' | 'boj' | 'boe'
  event_date date not null,
  announce_utc timestamptz,         -- exact announcement instant if known
  decision text,                    -- 'hike' | 'hold' | 'cut' | null (post-hoc)
  source_url text not null,
  notes text,
  unique (event_type, event_date)
);

-- 2. CFTC Commitments of Traders positions
create table if not exists cot_positions (
  contract_id text not null,        -- our id from cot_contracts.yaml
  report_date date not null,        -- 'as of' Tuesday
  released_on date not null,
  category text not null,
  long_pos double precision,
  short_pos double precision,
  spread_pos double precision,
  open_interest double precision,
  primary key (contract_id, report_date, category)
);

-- 3. Sovereign yield curve points
create table if not exists curve_points (
  curve_id text not null,           -- 'us_treasury_par', 'ecb_spot', ...
  d date not null,
  maturity_months int not null,
  yield_pct double precision not null,
  primary key (curve_id, d, maturity_months)
);

-- 4. Weekly physical fundamentals (energy inventories, etc.)
create table if not exists weekly_levels (
  series_id text not null references series_meta(id),
  week_end date not null,
  value double precision not null,
  primary key (series_id, week_end)
);

-- Enable Row-Level Security (RLS) on all four tables
alter table event_calendar enable row level security;
alter table cot_positions enable row level security;
alter table curve_points enable row level security;
alter table weekly_levels enable row level security;

-- Public read-only policies (public select on derived / public domain data)
create policy "Allow public select on event_calendar" on event_calendar for select using (true);
create policy "Allow public select on cot_positions" on cot_positions for select using (true);
create policy "Allow public select on curve_points" on curve_points for select using (true);
create policy "Allow public select on weekly_levels" on weekly_levels for select using (true);
