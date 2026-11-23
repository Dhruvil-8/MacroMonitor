import os
import re

components = [
    ("C01", "MoveDecomposerWaterfall", "/"),
    ("C02", "CountryMatrix", "/countries"),
    ("C03", "WorldChoropleth", "/countries"),
    ("C04", "MarketStateMap", "/"),
    ("C05", "RotationMap", "/sectors"),
    ("C06", "DriverSensitivityHeatmap", "/drivers"),
    ("C07", "RollingBetaChart", "/drivers"),
    ("C08", "DriverForestPlot", "/drivers"),
    ("C09", "DriverBivariateScatter", "/drivers"),
    ("C10", "VolatilityConeChart", "/risk"),
    ("C11", "DistributionHistogram", "/context/[id]"),
    ("C12", "ContextQqPlot", "/context/[id]"),
    ("C13", "TailRiskBarChart", "/risk"),
    ("C14", "AutocorrelationBarChart", "/risk"),
    ("C15", "UnderwaterDrawdownChart", "/risk"),
    ("C16", "PcaSynchronizationChart", "/risk"),
    ("C17", "DownsideCorrelationChart", "/risk"),
    ("C18", "HexbinJointDensityChart", "/risk"),
    ("C19", "RidgelineDensityChart", "/risk"),
    ("C20", "PercentileStrip", "/"),
    ("C21", "ZScoreBandChart", "/radar"),
    ("C22", "CorrelationHeatmap", "/cross-asset"),
    ("C23", "DivergencePanel", "/radar"),
    ("C24", "CusumChangeChart", "/radar"),
    ("C25", "CalendarHeatmap", "/context/[id]"),
    ("C26", "FanChart", "/risk"),
    ("C27", "NetLiquidityChart", "/liquidity"),
]

app_files = {}
for root, dirs, files in os.walk("web/src/app"):
    for f in files:
        if f.endswith(".tsx") and "charts" not in root:
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fh:
                app_files[path] = fh.read()

print("Checking where each of the 27 components is actually used across app pages:")
for code, comp, claimed_page in components:
    found_in = []
    for path, content in app_files.items():
        if comp in content:
            found_in.append(path)
    print(f"{code:4s} | {comp:26s} | Claimed: {claimed_page:16s} | Found in {len(found_in)} pages: {found_in}")
