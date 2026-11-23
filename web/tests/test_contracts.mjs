// web/tests/test_contracts.mjs
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { z } from "zod";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, "../public/data/v1");

// Schemas
const ManifestFileEntrySchema = z.object({
  path: z.string(),
  schema_version: z.string(),
  size_bytes: z.number(),
  as_of: z.string(),
});

const ManifestEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  manifest: z.object({
    datasets: z.array(ManifestFileEntrySchema),
  }),
});

const StateMapItemSchema = z.object({
  asset: z.string(),
  name: z.string(),
  group: z.string(),
  region: z.string(),
  trend_t: z.number(),
  vol_percentile: z.number(),
  quadrant: z.enum(["Orderly trend", "Violent trend", "Quiet range", "Choppy"]),
  momentum_20d: z.number(),
});

const StateMapEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: z.object({
    assets: z.array(StateMapItemSchema),
  }),
});

const VolLandscapeItemSchema = z.object({
  asset: z.string(),
  name: z.string(),
  group: z.string(),
  region: z.string(),
  vol10: z.number(),
  vol20: z.number(),
  vol60: z.number(),
  vol_trend: z.number(),
  vol20_percentile: z.number(),
  n: z.number(),
  n_eff: z.number(),
});

const VolLandscapeEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: z.object({
    landscape: z.array(VolLandscapeItemSchema),
  }),
});

const PulseStatSchema = z.object({
  id: z.string(),
  name: z.string(),
  value: z.number(),
  change_1d: z.number().nullable(),
  unit: z.string(),
  percentile: z.number().nullable(),
  sparkline: z.array(z.number()),
});

const PulseEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: z.object({
    headline: z.string(),
    risk_regime: z.string(),
    stats: z.array(PulseStatSchema),
    observations_count: z.number(),
  }),
});

const CompactSeriesEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: z.object({
    id: z.string(),
    name: z.string(),
    unit: z.string(),
    frequency: z.string(),
    d: z.array(z.string()),
    v: z.array(z.number().nullable()),
  }),
});

function runTests() {
  console.log("Verifying JSON contracts in:", dataDir);

  // 1. Manifest
  const manifestRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "manifest.json"), "utf8"));
  ManifestEnvelopeSchema.parse(manifestRaw);
  console.log("✓ manifest.json validated with Zod schema");

  // 2. Pulse
  const pulseRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "pulse.json"), "utf8"));
  PulseEnvelopeSchema.parse(pulseRaw);
  console.log("✓ pulse.json validated with Zod schema");

  // 3. State Map
  const stateRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "state_map.json"), "utf8"));
  StateMapEnvelopeSchema.parse(stateRaw);
  console.log("✓ state_map.json validated with Zod schema");

  // 4. Vol Landscape
  const volRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "vol_landscape.json"), "utf8"));
  VolLandscapeEnvelopeSchema.parse(volRaw);
  console.log("✓ vol_landscape.json validated with Zod schema");

  // 5. SPX Series
  const spxRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "series/spx.json"), "utf8"));
  CompactSeriesEnvelopeSchema.parse(spxRaw);
  console.log("✓ series/spx.json validated with Zod schema");

  // 6. Liquidity
  const liqRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "liquidity.json"), "utf8"));
  const LiquidityEnvelopeSchema = z.object({
    schema_version: z.string(),
    as_of: z.string(),
    code_version: z.string(),
    params_hash: z.string(),
    data: z.object({
      current_net_liquidity: z.number(),
      change_4w: z.number(),
      change_4w_pct: z.number(),
      change_13w: z.number(),
      change_13w_pct: z.number(),
      stablecoins_total_bn: z.number(),
      stablecoins_30d_pct: z.number(),
      dates: z.array(z.string()),
      net_liquidity: z.array(z.number()),
      fed_assets: z.array(z.number()),
      tga: z.array(z.number()),
      rrp: z.array(z.number()),
      spx_overlay: z.array(z.number()),
      rolling_corr_60d: z.array(z.number()),
      audit: z.object({
        formula: z.string(),
        parameters: z.record(z.string(), z.any()),
        series_used: z.array(z.string()),
        n: z.number(),
        n_eff: z.number(),
      }),
    }),
  });
  LiquidityEnvelopeSchema.parse(liqRaw);
  console.log("✓ liquidity.json validated with Zod schema");

  // 7. India Flows
  const indiaRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "india_flows.json"), "utf8"));
  const IndiaFlowsEnvelopeSchema = z.object({
    schema_version: z.string(),
    as_of: z.string(),
    code_version: z.string(),
    params_hash: z.string(),
    data: z.object({
      india_vix: z.number(),
      india_vix_percentile: z.number(),
      pcr: z.number(),
      fii_net_1d: z.number(),
      dii_net_1d: z.number(),
      fii_net_5d: z.number(),
      dii_net_5d: z.number(),
      fii_net_20d: z.number(),
      dii_net_20d: z.number(),
      fii_20d_percentile: z.number(),
      dii_20d_percentile: z.number(),
      delivery_pct_index: z.number(),
      delivery_pct_percentile: z.number(),
      flow_dates: z.array(z.string()),
      fii_daily: z.array(z.number()),
      dii_daily: z.array(z.number()),
      fii_cum_20d: z.array(z.number()),
      dii_cum_20d: z.array(z.number()),
      oi_buildup: z.array(
        z.object({
          symbol: z.string(),
          price_change_pct: z.number(),
          oi_change_pct: z.number(),
          buildup: z.enum(["Long Buildup", "Short Buildup", "Short Covering", "Long Unwinding", "Neutral"]),
          pcr: z.number(),
        })
      ),
      audit: z.object({
        formula: z.string(),
        parameters: z.record(z.string(), z.any()),
        series_used: z.array(z.string()),
        n: z.number(),
        n_eff: z.number(),
      }),
    }),
  });
  IndiaFlowsEnvelopeSchema.parse(indiaRaw);
  console.log("✓ india_flows.json validated with Zod schema");

  // 8. Financial Stress Composite
  const stressRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "stress.json"), "utf8"));
  const StressEnvelopeSchema = z.object({
    schema_version: z.string(),
    as_of: z.string(),
    code_version: z.string(),
    params_hash: z.string(),
    data: z.object({
      current_stress: z.number(),
      regime: z.string(),
      components: z.record(z.string(), z.number()),
      missing_components: z.array(z.string()),
      dates: z.array(z.string()),
      history: z.array(z.number()),
      episodes: z.array(
        z.object({
          name: z.string(),
          date: z.string(),
          stress: z.number(),
        })
      ),
      audit: z.object({
        formula: z.string(),
        parameters: z.record(z.string(), z.any()),
        series_used: z.array(z.string()),
        n: z.number(),
        n_eff: z.number(),
      }),
    }),
  });
  StressEnvelopeSchema.parse(stressRaw);
  console.log("✓ stress.json validated with Zod schema");

  // 9. Daily Briefing
  const briefingRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "briefing/latest.json"), "utf8"));
  const BriefingEnvelopeSchema = z.object({
    schema_version: z.string(),
    as_of: z.string(),
    code_version: z.string(),
    params_hash: z.string(),
    data: z.object({
      as_of: z.string(),
      headline: z.string(),
      executive_summary: z.string(),
      stress_score: z.number(),
      stress_regime: z.string(),
      sections: z.array(
        z.object({
          title: z.string(),
          content: z.string(),
        })
      ),
    }),
  });
  BriefingEnvelopeSchema.parse(briefingRaw);
  console.log("✓ briefing/latest.json validated with Zod schema");

  // 10. Move Decomposition
  const decompRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "decomposition/featured.json"), "utf8"));
  const DecompositionEnvelopeSchema = z.object({
    schema_version: z.string(),
    as_of: z.string(),
    code_version: z.string(),
    params_hash: z.string(),
    data: z.object({
      target: z.string(),
      target_id: z.string(),
      horizon_days: z.number(),
      lens: z.enum(["local", "usd"]),
      total_return: z.number(),
      contributions: z.record(z.string(), z.number()),
      intervals: z.record(z.string(), z.tuple([z.number(), z.number()])),
      residual: z.number(),
      betas: z.record(z.string(), z.number()),
      r2_in_sample: z.number(),
      r2_out_sample: z.number(),
      mostly_unexplained: z.boolean(),
      regularized: z.boolean(),
      n_obs: z.number(),
      window_dates: z.object({
        prior_start: z.string(),
        prior_end: z.string(),
        horizon_start: z.string(),
        horizon_end: z.string(),
      }),
      narrative: z.string(),
      history: z.array(
        z.object({
          date: z.string(),
          total_return: z.number(),
          contributions: z.record(z.string(), z.number()),
          residual: z.number(),
        })
      ).optional(),
    }),
  });
  DecompositionEnvelopeSchema.parse(decompRaw);
  console.log("✓ decomposition/featured.json validated with Zod schema");

  // 11. Country Layer (M5)
  const countriesRaw = JSON.parse(fs.readFileSync(path.join(dataDir, "countries.json"), "utf8"));
  const CountriesEnvelopeSchema = z.object({
    schema_version: z.string(),
    as_of: z.string(),
    code_version: z.string(),
    params_hash: z.string(),
    data: z.object({
      countries: z.array(
        z.object({
          code: z.string(),
          name: z.string(),
          index_id: z.string(),
          currency: z.string(),
          returns_local: z.record(z.string(), z.number()),
          returns_usd: z.record(z.string(), z.number()),
          returns_fx: z.record(z.string(), z.number()),
        })
      ),
      breadth_200d: z.number(),
      avg_corr_60d: z.number(),
      avg_corr_250d: z.number(),
      tracked_count: z.number(),
    }),
  });
  CountriesEnvelopeSchema.parse(countriesRaw);
  console.log("✓ countries.json validated with Zod schema");

  console.log("ALL ZOD CONTRACT TESTS PASSED SUCCESSFULLY!");
}

runTests();
