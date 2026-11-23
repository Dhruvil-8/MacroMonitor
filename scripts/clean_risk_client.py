with open("web/src/app/risk/RiskClientView.tsx", "r", encoding="utf-8") as f:
    text = f.read()

# Remove the inserted block
block = """      {/* Row 6: Non-Linear Joint Tail Density (C18) & Ridgeline Return Distributions (C19) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* C18: Hexbin Joint Tail Density */}
        <div className="lg:col-span-6">
          <ChartCard
            eyebrow="Non-Linear Tail Dependency (C18)"
            title="Hexbin Joint Tail Density"
            subtitle={`Bivariate empirical density of ${profile?.asset_name || "S&P 500"} vs US 10Y Yield`}
            asOf={asOf}
            methodData={{
              chartTitle: `Hexbin Joint Tail Density (C18) — ${profile?.asset_name || "S&P 500"} vs Macro Drivers`,
              whatItShows: "2D hexagonal binning of joint returns showing asymmetric tail co-dependence without assuming linear correlation.",
              definitionFormula: "\\\\hat{f}(x, y) = \\\\frac{1}{n b_x b_y} \\\\sum_{i=1}^n K\\\\left(\\\\frac{x - x_i}{b_x}\\\\right) K\\\\left(\\\\frac{y - y_i}{b_y}\\\\right)",
              symbolsDefined: [
                { symbol: "b_x, b_y", meaning: "Bandwidths along x and y coordinates" },
                { symbol: "K", meaning: "Hexagonal kernel weighting function" },
              ],
              parameters: { grid_size: "30x30", density_scale: "Empirical density mapped" },
              dataSources: [{ series: "SPX & US10Y", source: "Daily aligned changes", frequency: "Daily" }],
              sample: { n: 250, nEff: 250, dateRange: "Trailing 250 trading days" },
              interpretationNotes: [
                "Hexagonal clusters in the lower quadrants highlight simultaneous extreme joint sell-offs or flight-to-safety episodes.",
              ],
              reproduce: {
                functionName: "compute_bivariate_tail_density",
                filePath: "pipeline/src/radar/metrics/risk.py",
                testPath: "pipeline/tests/unit/test_risk.py",
              },
            }}
            auditData={{
              metricName: "Hexbin Joint Tail Density",
              formula: "Bivariate 2D empirical histogram / density",
              formulaPlain: "Hexagonal binning of daily asset return vs macro factor changes.",
              parameters: { grid: "hexagonal", lookback: "250D" },
              inputs: [
                { id: "spx", name: "S&P 500", source: "Market Returns", unit: "%" },
                { id: "us10y", name: "US 10Y Yield", source: "FRED", unit: "bps" },
              ],
              window: "Trailing 250 trading days",
              currentValue: "Asymmetric tail cluster",
            }}
            onOpenAudit={handleOpenAudit}
            onOpenMethod={handleOpenMethod}
          >
            <HexbinJointDensityChart
              xLabel={`${profile?.asset_name || "S&P 500"} Daily Return (%)`}
              yLabel="US 10Y Yield Change (bps)"
            />
          </ChartCard>
        </div>

        {/* C19: Ridgeline Return Distributions */}
        <div className="lg:col-span-6">
          <ChartCard
            eyebrow="Distribution Shape Comparison (C19)"
            title="Ridgeline Return Distributions"
            subtitle="Overlapping KDE curves across 8 key asset classes highlighting asymmetry and current position"
            asOf={asOf}
            methodData={{
              chartTitle: "Cross-Market Ridgeline Return Distributions (C19)",
              whatItShows: "Stacked density curves across global equity markets and commodities, with pin markers showing today's 1-day return.",
              definitionFormula: "\\\\hat{f}_m(x) = \\\\frac{1}{nh} \\\\sum_{i=1}^n \\\\frac{1}{\\\\sqrt{2\\\\pi}} e^{-\\\\frac{(x - r_{m,i})^2}{2h^2}}",
              symbolsDefined: [
                { symbol: "r_{m,i}", meaning: "Daily return for market m" },
                { symbol: "h", meaning: "Silverman bandwidth" },
              ],
              parameters: { markets: "8 core international assets", span: "-5% to +5%" },
              dataSources: [{ series: "8 Launch Indexes", source: "Exchange Returns", frequency: "Daily" }],
              sample: { n: 250, nEff: 250, dateRange: "Full available history" },
              interpretationNotes: [
                "Broader ridges indicate higher volatility regimes (e.g. Crude Oil vs FTSE 100).",
                "Pin markers shifted to the tails reveal which markets experienced abnormal 1-day shocks.",
              ],
              reproduce: {
                functionName: "compute_ridgeline_densities",
                filePath: "pipeline/src/radar/metrics/risk.py",
                testPath: "pipeline/tests/unit/test_risk.py",
              },
            }}
            auditData={{
              metricName: "Cross-Asset Ridgeline Density Stack",
              formula: "Stacked Kernel Density Estimates across markets",
              formulaPlain: "Multi-market KDE comparison of daily returns with current realization pins.",
              parameters: { n_markets: 8 },
              inputs: [{ id: "universe", name: "Global Equity & Commodity Core", source: "Exchange Closes", unit: "%" }],
              window: "250-Day lookback",
              currentValue: "8 Market Profiles Mapped",
            }}
            onOpenAudit={handleOpenAudit}
            onOpenMethod={handleOpenMethod}
          >
            <RidgelineDensityChart />
          </ChartCard>
        </div>
      </div>
"""

# Replace all occurrences with empty string
text = text.replace(block, "")

# Now insert it ONCE right before {/* Drawers */}
target = "{/* Drawers */}"
if target in text:
    text = text.replace(target, block + "\n      " + target)

with open("web/src/app/risk/RiskClientView.tsx", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated RiskClientView.tsx cleanly")
