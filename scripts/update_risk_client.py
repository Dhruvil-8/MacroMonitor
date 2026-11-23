import re

with open('web/src/app/risk/RiskClientView.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update imports
old_imports = '''import { VolatilityConeChart } from "@/components/charts/VolatilityConeChart";
import { TailRiskBarChart } from "@/components/charts/TailRiskBarChart";'''

new_imports = '''import { VolatilityConeChart } from "@/components/charts/VolatilityConeChart";
import { DistributionHistogram } from "@/components/charts/DistributionHistogram";
import { ContextQqPlot } from "@/components/charts/ContextQqPlot";
import { TailRiskBarChart } from "@/components/charts/TailRiskBarChart";'''

content = content.replace(old_imports, new_imports)

old_client_import = '''import { loadRisk, loadStress } from "@/lib/data/client";
import { RiskEnvelope, AssetRiskProfile, StressEnvelope } from "@/lib/data/schemas";'''

new_client_import = '''import { loadRisk, loadStress, loadContext } from "@/lib/data/client";
import { RiskEnvelope, AssetRiskProfile, StressEnvelope, ContextData } from "@/lib/data/schemas";'''

content = content.replace(old_client_import, new_client_import)

# 2. Add contextData state and useEffect
state_target = '''  const [selectedAssetId, setSelectedAssetId] = useState<string>("spx");'''

new_state = '''  const [selectedAssetId, setSelectedAssetId] = useState<string>("spx");
  const [contextData, setContextData] = useState<ContextData | null>(null);

  useEffect(() => {
    async function fetchContext() {
      if (!selectedAssetId) return;
      try {
        const res = await loadContext(selectedAssetId);
        if (res.status !== "error") {
          setContextData(res.data.data);
        }
      } catch (err) {
        console.error("Failed to load context for asset", selectedAssetId, err);
      }
    }
    fetchContext();
  }, [selectedAssetId]);'''

content = content.replace(state_target, new_state)

# 3. Insert Row 2 (Distribution Modeling C11 & C12) right before Row 2 (Tail Risk C13)
# Currently lines 256 has:
#         </ChartCard>
#       )}
# 
# 
#       {/* Row 6: Non-Linear Joint Tail Density (C18) & Ridgeline Return Distributions (C19) */}

row2_code = '''
      {/* Row 2: Distribution Modeling (C11) & Normality Diagnostic (C12) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* C11: Histogram & KDE */}
        <div className="lg:col-span-7">
          <ChartCard
            eyebrow="Distribution Modeling (C11)"
            title={`${profile?.asset_name || "S&P 500"} Return Distribution`}
            subtitle="Empirical histogram overlaid with Silverman's Gaussian KDE and Normal N(μ, σ²)"
            asOf={asOf}
            methodData={{
              chartTitle: `Return Distribution & Gaussian KDE (C11) — ${profile?.asset_name || "S&P 500"}`,
              whatItShows: "Empirical daily return frequency compared against a non-parametric Gaussian Kernel Density Estimate (Silverman's rule) and classical Gaussian normal bell curve.",
              definitionFormula: "h = 0.9 \\\\cdot \\\\min\\\\left(\\\\hat{\\\\sigma}, \\\\frac{\\\\text{IQR}}{1.34}\\\\right) n^{-1/5}, \\\\quad \\\\hat{f}(x) = \\\\frac{1}{nh\\\\sqrt{2\\\\pi}} \\\\sum_{i=1}^n e^{-\\\\frac{(x - r_i)^2}{2h^2}}",
              symbolsDefined: [
                { symbol: "h", meaning: "Silverman's optimal bandwidth for Gaussian kernel" },
                { symbol: "IQR", meaning: "Interquartile range (75th percentile - 25th percentile)" },
              ],
              parameters: {
                bandwidth_rule: "Silverman's rule of thumb",
                num_bins: 25,
                grid_points: 60,
              },
              dataSources: [{ series: selectedAssetId, source: "Primary Exchange Feeds", frequency: "Daily" }],
              sample: { n: 250, nEff: 250, dateRange: "Full available history" },
              interpretationNotes: [
                "Gap between the teal KDE curve and lavender dashed normal curve reveals empirical fat tails and skewness.",
                "The orange vertical marker indicates where today's realization sits relative to history.",
              ],
              reproduce: {
                functionName: "compute_kde_and_normal_overlay",
                filePath: "pipeline/src/radar/metrics/context.py",
                testPath: "pipeline/tests/unit/test_context.py",
              },
            }}
            auditData={{
              metricName: `${profile?.asset_name || "S&P 500"} Distribution Histogram`,
              formula: "Empirical histogram + Gaussian KDE + Normal PDF",
              formulaPlain: "25-bin empirical distribution of daily stationary returns with KDE smoothing.",
              parameters: { num_bins: 25, bandwidth: "Silverman" },
              inputs: [{ id: selectedAssetId, name: profile?.asset_name || "S&P 500", source: "Primary Feeds", unit: "%" }],
              window: "Full sample expanding",
              currentValue: contextData?.current_value ?? profile?.volatility_cone?.[0]?.current ?? 0,
              percentile: contextData?.percentile ?? 50,
            }}
            onOpenAudit={handleOpenAudit}
            onOpenMethod={handleOpenMethod}
          >
            {contextData ? (
              <DistributionHistogram
                bins={contextData.histogram_bins}
                counts={contextData.histogram_counts}
                kdeGrid={contextData.kde_grid}
                kdeDensity={contextData.kde_density}
                normalDensity={contextData.normal_density}
                currentValue={contextData.current_value}
                percentile={contextData.percentile}
                unit={contextData.unit}
              />
            ) : (
              <div className="h-[280px] flex items-center justify-center font-mono text-xs text-[var(--tx-2)]">
                Loading Distribution Data for {profile?.asset_name || selectedAssetId.toUpperCase()}...
              </div>
            )}
          </ChartCard>
        </div>

        {/* C12: QQ Plot */}
        <div className="lg:col-span-5">
          <ChartCard
            eyebrow="Normality Diagnostic (C12)"
            title="Quantile-Quantile (QQ) Plot"
            subtitle="Empirical standardized quantiles vs Theoretical Gaussian Normal quantiles"
            asOf={asOf}
            methodData={{
              chartTitle: `Quantile-Quantile Plot (C12) — ${profile?.asset_name || "S&P 500"}`,
              whatItShows: "Empirical quantiles plotted against theoretical normal quantiles; departures from the 45-degree diagonal line immediately expose heavy tails and skewness.",
              definitionFormula: "q_{\\\\text{theoretical}} = \\\\Phi^{-1}(p), \\\\quad q_{\\\\text{empirical}} = \\\\frac{\\\\text{percentile}(r, p) - \\\\hat{\\\\mu}}{\\\\hat{\\\\sigma}}",
              symbolsDefined: [
                { symbol: "Phi^-1", meaning: "Standard normal inverse cumulative distribution function (probit)" },
                { symbol: "p", meaning: "Quantile probabilities spanning 0.01 to 0.99" },
              ],
              parameters: {
                num_quantiles: 50,
                bounds: "0.01 to 0.99 (excluding asymptotic limits)",
              },
              dataSources: [{ series: selectedAssetId, source: "Primary Exchange Feeds", frequency: "Daily" }],
              sample: { n: 250, nEff: 250, dateRange: "Full available history" },
              interpretationNotes: [
                "Points sagging below the diagonal in the left tail indicate negative crash risk heavier than a normal distribution.",
                "S-shaped deviations indicate excess kurtosis (leptokurtic distribution).",
              ],
              reproduce: {
                functionName: "compute_qq_quantiles",
                filePath: "pipeline/src/radar/metrics/context.py",
                testPath: "pipeline/tests/unit/test_context.py",
              },
            }}
            auditData={{
              metricName: "Normal Quantile-Quantile Plot",
              formula: "norm.ppf(p) vs standardized sample quantiles",
              formulaPlain: "50 quantile points compared against theoretical normal distribution.",
              parameters: { points: 50 },
              inputs: [{ id: selectedAssetId, name: profile?.asset_name || "S&P 500", source: "Primary Feeds", unit: "sigma" }],
              window: "Full sample",
              currentValue: `${profile?.distribution?.skewness ?? 0} skew, ${profile?.distribution?.excess_kurtosis ?? 0} kurtosis`,
            }}
            onOpenAudit={handleOpenAudit}
            onOpenMethod={handleOpenMethod}
          >
            {contextData || (profile?.distribution?.qq_theoretical && profile.distribution.qq_theoretical.length > 0) ? (
              <ContextQqPlot
                theoretical={contextData?.qq_theoretical || profile!.distribution.qq_theoretical}
                empirical={contextData?.qq_empirical || profile!.distribution.qq_empirical}
                assetName={profile?.asset_name || selectedAssetId.toUpperCase()}
              />
            ) : (
              <div className="h-[280px] flex items-center justify-center font-mono text-xs text-[var(--tx-2)]">
                Loading QQ Normality Diagnostics...
              </div>
            )}
          </ChartCard>
        </div>
      </div>
'''

target_anchor = '''        <VolatilityConeChart coneData={profile.volatility_cone} assetName={profile.asset_name} />
        </ChartCard>
      )}'''

content = content.replace(target_anchor, target_anchor + "\n" + row2_code)

with open('web/src/app/risk/RiskClientView.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated RiskClientView.tsx successfully!")
