"use client";

import { useMemo, useState } from "react";
import { Check, Code2, Copy, Download, ExternalLink, X } from "lucide-react";
import { useRadarStore } from "@/store/useRadarStore";

function escXml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function buildShareCardSvg(opts: {
  title: string;
  subtitle: string;
  headlineMetric: string;
  percentileSummary: string;
  asOf: string;
  n: number | string;
  nEff: number | string;
}): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="440" viewBox="0 0 960 440">
  <rect width="960" height="440" rx="10" fill="#07090D" stroke="#1D2633" stroke-width="2"/>
  <rect x="32" y="28" width="896" height="384" rx="8" fill="#0C1017" stroke="#1D2633"/>
  <text x="64" y="76" fill="#38BDF8" font-family="system-ui, -apple-system, sans-serif" font-size="14" font-weight="600" letter-spacing="1.2">MACROMONITOR · MACRO STATISTICS</text>
  <text x="64" y="122" fill="#E8EDF5" font-family="system-ui, -apple-system, sans-serif" font-size="30" font-weight="700">${escXml(opts.title.slice(0, 56))}</text>
  <text x="64" y="156" fill="#8C9BB0" font-family="system-ui, -apple-system, sans-serif" font-size="16">${escXml(opts.subtitle.slice(0, 92))}</text>
  <rect x="64" y="190" width="832" height="122" rx="6" fill="#07090D" stroke="#1D2633"/>
  <text x="88" y="236" fill="#8C9BB0" font-family="monospace" font-size="13">KEY OBSERVATION / SUMMARY STATISTIC</text>
  <text x="88" y="274" fill="#2DD4BF" font-family="monospace" font-size="22" font-weight="600">${escXml(opts.headlineMetric.slice(0, 68))}</text>
  <text x="88" y="300" fill="#E8EDF5" font-family="monospace" font-size="13">${escXml(opts.percentileSummary.slice(0, 96))}</text>
  <line x1="64" y1="342" x2="896" y2="342" stroke="#1D2633"/>
  <text x="64" y="378" fill="#6F7C90" font-family="monospace" font-size="13">Source: MacroMonitor · Descriptive statistics only · As of ${escXml(opts.asOf)} · N=${escXml(String(opts.n))}, N_eff=${escXml(String(opts.nEff))}</text>
</svg>`;
}

export function ShareEmbedModal() {
  const shareTarget = useRadarStore((s) => s.shareTarget);
  const closeShare = useRadarStore((s) => s.closeShare);
  const [copiedEmbed, setCopiedEmbed] = useState(false);

  const svgMarkup = useMemo(() => {
    if (!shareTarget) return "";
    return buildShareCardSvg(shareTarget);
  }, [shareTarget]);

  if (!shareTarget) return null;

  const origin = typeof window !== "undefined" ? window.location.origin : "https://macromonitor.org";
  const sp = new URLSearchParams({ embed: shareTarget.chartKey, ...(shareTarget.queryParams ?? {}) });
  const embedUrl = `${origin}/?${sp.toString()}`;
  const iframeSnippet = `<iframe src="${embedUrl}" width="780" height="460" style="border:1px solid #1D2633;border-radius:6px;" title="${shareTarget.title} — MacroMonitor" loading="lazy"></iframe>`;

  const handleDownloadSvg = () => {
    const blob = new Blob([svgMarkup], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `gmr-${shareTarget.chartKey}-${shareTarget.asOf}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadPng = () => {
    const img = new Image();
    const svgBlob = new Blob([svgMarkup], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(svgBlob);
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = 960;
      canvas.height = 440;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        canvas.toBlob((pngBlob) => {
          if (pngBlob) {
            const pngUrl = URL.createObjectURL(pngBlob);
            const a = document.createElement("a");
            a.href = pngUrl;
            a.download = `gmr-${shareTarget.chartKey}-${shareTarget.asOf}.png`;
            a.click();
            URL.revokeObjectURL(pngUrl);
          }
        }, "image/png");
      }
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  const handleCopyEmbed = async () => {
    try {
      await navigator.clipboard.writeText(iframeSnippet);
      setCopiedEmbed(true);
      setTimeout(() => setCopiedEmbed(false), 1800);
    } catch {
      setCopiedEmbed(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Share Card & Embeddable Widget"
    >
      <div className="w-full max-w-2xl rounded-[8px] border border-[#1d2633] bg-[#0c1017] p-5 flex flex-col gap-4 max-h-[90vh] overflow-y-auto shadow-2xl">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-[#e8edf5]">
              Share Statistical Card &amp; Embed Widget
            </h3>
            <p className="text-xs text-[#8c9bb0] mt-0.5">
              Static attribution card and chrome-free iframe widget with permanent As-Of provenance.
            </p>
          </div>
          <button
            type="button"
            onClick={closeShare}
            aria-label="Close Share dialog"
            className="p-1.5 rounded-[6px] text-[#8c9bb0] hover:text-[#e8edf5] cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Visual SVG Card Preview */}
        <div
          className="rounded-[6px] overflow-hidden border border-[#1d2633] bg-[#07090d]"
          dangerouslySetInnerHTML={{ __html: svgMarkup }}
        />

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleDownloadPng}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-[6px] bg-[#38bdf8] text-[#07090d] hover:bg-[#38bdf8]/90 cursor-pointer font-semibold"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download PNG Card</span>
          </button>
          <button
            type="button"
            onClick={handleDownloadSvg}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-[6px] border border-[#1d2633] text-[#e8edf5] hover:bg-[#121823] cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download SVG</span>
          </button>
          <a
            href={embedUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-[6px] border border-[#1d2633] text-[#8c9bb0] hover:text-[#e8edf5] hover:bg-[#121823]"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            <span>Preview Link</span>
          </a>
        </div>

        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-xs font-medium text-[#8c9bb0]">
            <span className="inline-flex items-center gap-1.5">
              <Code2 className="w-3.5 h-3.5" />
              <span>Iframe Embed Code</span>
            </span>
            <button
              type="button"
              onClick={handleCopyEmbed}
              className="inline-flex items-center gap-1 text-[#38bdf8] hover:underline cursor-pointer"
            >
              {copiedEmbed ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copiedEmbed ? "Copied" : "Copy HTML"}</span>
            </button>
          </div>
          <pre className="p-2.5 rounded-[6px] border border-[#1d2633] bg-[#07090d] text-[11px] font-mono text-[#e8edf5] overflow-x-auto">
            {iframeSnippet}
          </pre>
        </div>
      </div>
    </div>
  );
}
