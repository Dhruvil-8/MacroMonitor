"use client";

import { useEffect, useMemo, useState } from "react";
import { downloadCsv } from "@/lib/formatters";
import { loadAnaloguePreset, loadAnaloguesEvaluation, loadAnaloguesIndex } from "@/lib/data/client";
import {
  AnaloguePresetDetailData,
  AnaloguesEvaluationData,
  AnaloguesPresetsData,
} from "@/lib/data/schemas";
import { useRadarStore } from "@/store/useRadarStore";
import { ChartPanel } from "@/components/insight/ChartPanel";
import { AnalogueMatchTable } from "@/components/v2/AnalogueMatchTable";
import { FanChart } from "@/components/v2/FanChart";
import { VerdictCallout } from "@/components/v2/VerdictCallout";

export default function AnaloguesPage() {
  const presetId = useRadarStore((s) => s.analoguePreset);
  const setPresetId = useRadarStore((s) => s.setAnaloguePreset);

  const [indexData, setIndexData] = useState<{
    presets: AnaloguesPresetsData;
    evaluation: AnaloguesEvaluationData;
  } | null>(null);
  const [detail, setDetail] = useState<AnaloguePresetDetailData | null>(null);
  const [targetAsset, setTargetAsset] = useState("SPX");
  const [selectedRank, setSelectedRank] = useState(1);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([loadAnaloguesIndex(), loadAnaloguesEvaluation()])
      .then(([pRes, eRes]) => {
        if (pRes.status !== "error" && eRes.status !== "error") {
          setIndexData({
            presets: pRes.data,
            evaluation: eRes.data,
          });
        } else {
          setError(pRes.status === "error" ? pRes.error : (eRes as any).error);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    loadAnaloguePreset(presetId)
      .then((res) => {
        if (res.status !== "error") {
          setDetail(res.data);
          setSelectedRank(1);
          setTargetAsset(presetId === "india_flow_stress" ? "NIFTY50" : "SPX");
        } else {
          setError(res.error);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [presetId]);

  const availableAssets = useMemo(() => {
    if (!detail) return ["SPX"];
    return Array.from(new Set(detail.outcome_fans.map((f) => f.asset)));
  }, [detail]);

  const assetFans = useMemo(
    () => (detail ? detail.outcome_fans.filter((f) => f.asset === targetAsset) : []),
    [detail, targetAsset]
  );

  if (error) {
    return (
      <div className="rounded-[6px] border border-rose-500 bg-[#0c1017] p-5 text-sm text-rose-400">
        Failed to load historical analogue engine: {error}
      </div>
    );
  }

  if (!indexData || !detail) {
    return (
      <div className="h-96 rounded-[6px] border border-[#1d2633] bg-[#0c1017] animate-pulse flex items-center justify-center font-mono text-xs text-[#6f7c90]">
        Loading Ledoit-Wolf Shrinkage Analogue Engine...
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12">
      <div className="flex flex-col gap-1 pb-2 border-b border-[#1d2633]">
        <div className="text-[10px] font-mono uppercase tracking-widest text-[#6f7c90]">
          Pattern Recognition & Quantitative Analogue Search
        </div>
        <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-[#e8edf5]">
          Historical Analogue Engine &amp; Walk-Forward Calibration
        </h1>
        <p className="text-xs sm:text-sm text-[#8c9bb0]">
          Ledoit-Wolf shrinkage Mahalanobis similarity over strictly expanding standardized features, 20-day declustering, and honest out-of-sample evaluation.
        </p>
      </div>

      {/* Preset Selector Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {indexData.presets.presets.map((p) => {
          const active = p.id === presetId;
          return (
            <button
              key={p.id}
              type="button"
              onClick={() => setPresetId(p.id)}
              className={`text-left rounded-[6px] border p-3.5 transition-colors cursor-pointer flex flex-col justify-between gap-2 ${active
                ? "border-[#38bdf8] bg-[#38bdf8]/10"
                : "border-[#1d2633] bg-[#0c1017] hover:border-[#6f7c90]"
                }`}
            >
              <div>
                <div className="text-xs font-semibold text-[#e8edf5]">{p.name}</div>
                <div className="text-[11px] text-[#8c9bb0] mt-1 line-clamp-2">{p.rationale}</div>
              </div>
              <div className="text-[11px] font-mono tabular-nums text-[#6f7c90]">
                Top: {p.top_match_date ?? "—"} (D={p.top_match_distance?.toFixed(2) ?? "—"})
              </div>
            </button>
          );
        })}
      </div>

      {/* Honest Walk-Forward Verdict Banner */}
      <VerdictCallout
        verdictClass={detail.evaluation.verdict_class}
        verdictText={detail.evaluation.verdict_text}
        nEval={detail.evaluation.n_eval}
        spearmanRho={detail.evaluation.spearman_rho}
        iqrCoverage={detail.evaluation.iqr_coverage}
        maeSkill={detail.evaluation.mae_skill}
        targetAsset={detail.evaluation.target_asset || "SPX"}
      />

      {/* Conditional Outcome Fan ChartPanel */}
      <ChartPanel
        title={`${detail.name} — Conditional Outcomes vs. Unconditional Base Rate`}
        subtitle={`${detail.rationale} (${detail.matches.length} declustered historical episodes separated by >= ${detail.decluster_gap_days} trading days)`}
        infoText="Shaded regions show the 10th–90th and 25th–75th percentiles of subsequent returns after the top 15 historical matches. Dashed amber line is the unconditional base-rate median over all eligible history."
        insights={[
          assetFans[1]
            ? `${assetFans[1].asset} ${assetFans[1].horizon} match median: ${assetFans[1].cond_p50 >= 0 ? "+" : ""}${assetFans[1].cond_p50}${assetFans[1].unit} vs base rate ${assetFans[1].base_p50 >= 0 ? "+" : ""}${assetFans[1].base_p50}${assetFans[1].unit} (90% CI diff [${assetFans[1].diff_ci90_low}, ${assetFans[1].diff_ci90_high}]).`
            : `Query date ${detail.query_date} using N_indep = ${detail.n_indep} effective independent dimensions.`,
          `Out-of-sample walk-forward verdict (${detail.evaluation.n_eval} monthly dates): MAE skill ${(detail.evaluation.mae_skill * 100).toFixed(1)}%, IQR empirical coverage ${(detail.evaluation.iqr_coverage * 100).toFixed(1)}%.`,
        ]}
        asOf={detail.as_of}
        windowLabel={`1M / 3M / 6M Forward (${detail.matches.length} Declustered Episodes)`}
        sourcesLabel="MacroMonitor Expanding Standardizer"
        n={detail.matches.length}
        nEff={detail.matches.length}
        embedKey="analogues"
        headerRightExtra={
          <div className="flex flex-wrap items-center gap-1">
            {availableAssets.map((a) => (
              <button
                key={a}
                type="button"
                onClick={() => setTargetAsset(a)}
                className={`px-2 py-1 text-[11px] font-mono rounded-[4px] border cursor-pointer ${targetAsset === a
                  ? "border-[#38bdf8] bg-[#38bdf8]/10 text-[#e8edf5]"
                  : "border-[#1d2633] text-[#6f7c90]"
                  }`}
              >
                {a}
              </button>
            ))}
          </div>
        }
        audit={{
          title: `${detail.name} Mahalanobis Analogue Engine`,
          formula: "D(x_t, x_s) = sqrt((x_t - x_s)^T Sigma_LW^{-1} (x_t - x_s))",
          parameters: {
            n_features: detail.n_features,
            n_indep: detail.n_indep,
            shrinkage_delta: detail.shrinkage_delta,
            decluster_gap_days: detail.decluster_gap_days,
            top_k: detail.matches.length,
          },
          inputs: detail.feature_labels.map((lbl) => ({ id: lbl, lastUpdate: detail.query_date })),
          percentileSummary: detail.evaluation.verdict_text,
          n: detail.matches.length,
          nEff: detail.matches.length,
          asOf: detail.as_of,
          codeVersion: detail.code_version,
          paramsHash: detail.params_hash ?? "",
          specAnchor: "analogues",
        }}
        onDownloadCsv={() => {
          downloadCsv(
            `analogues-${detail.preset_id}-${detail.as_of}.csv`,
            ["rank", "date", "distance", "similarity_score", "spx_fwd_3m", "spx_mdd_3m", "us10y_fwd_3m"],
            detail.matches.map((m) => [m.rank, m.date, m.distance, m.similarity_score, m.spx_fwd_3m, m.spx_mdd_3m, m.us10y_fwd_3m])
          );
        }}
        getTableData={() => ({
          headers: ["Asset", "Horizon", "Matches N", "Analogue p50", "Base p50", "90% CI Diff", "Cond MDD p50"],
          rows: detail.outcome_fans.map((f) => [
            f.asset,
            f.horizon,
            f.n_matches,
            `${f.cond_p50}${f.unit}`,
            `${f.base_p50}${f.unit}`,
            `[${f.diff_ci90_low}, ${f.diff_ci90_high}]`,
            f.cond_mdd_p50 !== null ? `${f.cond_mdd_p50}%` : "—",
          ]),
        })}
      >
        <FanChart fans={detail.outcome_fans} selectedAsset={targetAsset} />
      </ChartPanel>

      {/* Feature Vector & Match List */}
      <section className="rounded-[6px] border border-[#1d2633] bg-[#0c1017] p-4 sm:p-5 flex flex-col gap-3">
        <div>
          <h2 className="text-sm font-semibold text-[#e8edf5]">
            Declustered Historical Matches (Click Row to Compare Features)
          </h2>
          <p className="text-xs text-[#6f7c90] mt-0.5">
            Minimum {detail.decluster_gap_days} trading days separation enforced between any two selected episodes.
          </p>
        </div>
        <AnalogueMatchTable detail={detail} selectedRank={selectedRank} onSelectRank={setSelectedRank} />
      </section>

      {/* Walk-Forward Subperiod Evaluation Table */}
      <section className="rounded-[6px] border border-[#1d2633] bg-[#0c1017] p-4 sm:p-5 flex flex-col gap-3">
        <div>
          <h2 className="text-sm font-semibold text-[#e8edf5]">
            Out-of-Sample Walk-Forward Sub-Period Breakdown (All Presets)
          </h2>
          <p className="text-xs text-[#6f7c90] mt-0.5">
            Calibration threshold: MAE skill &gt; +{(indexData.evaluation.skill_threshold * 100).toFixed(0)}% and N_eval ≥ {indexData.evaluation.min_eval_points}.
          </p>
        </div>
        <div className="overflow-x-auto rounded-[6px] border border-[#1d2633]">
          <table className="w-full text-xs border-collapse">
            <thead className="bg-[#07090d] border-b border-[#1d2633] text-[#6f7c90] text-left">
              <tr>
                <th className="py-2 px-3 font-medium">Preset</th>
                <th className="py-2 px-3 font-medium">Target</th>
                <th className="py-2 px-3 font-medium text-right">N_eval</th>
                <th className="py-2 px-3 font-medium text-right">Spearman ρ</th>
                <th className="py-2 px-3 font-medium text-right">IQR Coverage</th>
                <th className="py-2 px-3 font-medium text-right">MAE Skill</th>
                <th className="py-2 px-3 font-medium">Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1d2633] font-mono tabular-nums">
              {indexData.evaluation.presets.map((row) => (
                <tr key={row.preset_id} className={row.preset_id === presetId ? "bg-[#38bdf8]/10" : ""}>
                  <td className="py-1.5 px-3 font-sans font-medium text-[#e8edf5]">{row.name}</td>
                  <td className="py-1.5 px-3 text-[#8c9bb0]">{row.target_asset}</td>
                  <td className="py-1.5 px-3 text-right">{row.n_eval}</td>
                  <td className="py-1.5 px-3 text-right">{row.spearman_rho >= 0 ? "+" : ""}{row.spearman_rho.toFixed(3)}</td>
                  <td className="py-1.5 px-3 text-right">{(row.iqr_coverage * 100).toFixed(1)}%</td>
                  <td className="py-1.5 px-3 text-right">{row.mae_skill >= 0 ? "+" : ""}{(row.mae_skill * 100).toFixed(1)}%</td>
                  <td className="py-1.5 px-3 font-sans text-[#8c9bb0]">
                    {row.verdict_class === "improvement" ? "Modest skill improvement" : "No reliable improvement"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
