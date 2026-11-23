import { AlertCircle, CheckCircle2 } from "lucide-react";

interface VerdictCalloutProps {
  verdictClass: "improvement" | "no_reliable_improvement" | string;
  verdictText: string;
  nEval: number;
  spearmanRho: number;
  iqrCoverage: number;
  maeSkill: number;
  targetAsset?: string;
}

export function VerdictCallout({
  verdictClass,
  verdictText,
  nEval,
  spearmanRho,
  iqrCoverage,
  maeSkill,
  targetAsset = "SPX",
}: VerdictCalloutProps) {
  const improved = verdictClass === "improvement";
  return (
    <div
      className={`rounded-[6px] border p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
        improved
          ? "border-[#2dd4bf]/40 bg-[#2dd4bf]/5"
          : "border-[#f59e0b]/40 bg-[#f59e0b]/5"
      }`}
    >
      <div className="flex items-start gap-2.5">
        {improved ? (
          <CheckCircle2 className="w-4 h-4 text-[#2dd4bf] mt-0.5 shrink-0" />
        ) : (
          <AlertCircle className="w-4 h-4 text-[#f59e0b] mt-0.5 shrink-0" />
        )}
        <div className="text-xs leading-relaxed text-[#e8edf5]">
          <span className="font-semibold">Walk-Forward Calibration ({targetAsset} 3M): </span>
          <span>{verdictText}</span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 font-mono tabular-nums text-[11px] text-[#8c9bb0] shrink-0">
        <span>N_eval = {nEval}</span>
        <span>·</span>
        <span>Spearman ρ = {spearmanRho >= 0 ? "+" : ""}{spearmanRho.toFixed(3)}</span>
        <span>·</span>
        <span>IQR Cov = {(iqrCoverage * 100).toFixed(1)}% (target 50%)</span>
        <span>·</span>
        <span>MAE Skill = {maeSkill >= 0 ? "+" : ""}{(maeSkill * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}
