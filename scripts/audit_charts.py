import os, re, glob

# Read charts/page.tsx
with open('web/src/app/charts/page.tsx', 'r', encoding='utf-8') as f:
    text = f.read()

# Pattern for charts
chart_blocks = re.findall(r'\{\s*id:\s*"([^"]+)",\s*code:\s*"([^"]+)",\s*name:\s*"([^"]+)",\s*category:\s*"([^"]+)",\s*idea:\s*"([^"]+)",\s*insight:\s*"([^"]+)",\s*deployedPage:\s*"([^"]+)"', text)

print(f"Total charts in catalogue: {len(chart_blocks)}\n")

# Map code to expected component name
comp_names = {
    'C01': 'MoveDecomposerWaterfall',
    'C02': 'ReturnMatrixHeatmap',
    'C03': 'WorldChoropleth',
    'C04': 'MarketStateMap',
    'C05': 'RotationMap',
    'C06': 'DriverSensitivityHeatmap',
    'C07': 'RollingBetaChart',
    'C08': 'DriverForestPlot',
    'C09': 'DriverBivariateScatter',
    'C10': 'VolatilityConeChart',
    'C11': 'DistributionHistogram',
    'C12': 'ContextQqPlot',
    'C13': 'TailRiskBarChart',
    'C14': 'AutocorrelationBarChart',
    'C15': 'UnderwaterDrawdownChart',
    'C16': 'PcaSynchronizationChart',
    'C17': 'DownsideCorrelationChart',
    'C18': 'HexbinJointDensityChart',
    'C19': 'RidgelineDensityChart',
    'C20': 'PercentileStrip',
    'C21': 'ZScoreBandChart',
    'C22': 'CorrelationHeatmap',
    'C23': 'DivergencePanel',
    'C24': 'CusumChangeChart',
    'C25': 'CalendarHeatmap',
    'C26': 'FanChart',
    'C27': 'NetLiquidityChart'
}

all_tsx = glob.glob('web/src/**/*.tsx', recursive=True)
all_tsx = [p.replace('\\', '/') for p in all_tsx if 'charts/page.tsx' not in p.replace('\\', '/')]

results = []
for cid, code, name, cat, idea, insight, deployed in chart_blocks:
    comp = comp_names.get(code, '')
    mounted_in = []
    for fpath in all_tsx:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
            content = fp.read()
            if comp and f"<{comp}" in content:
                mounted_in.append(fpath)
    
    # Check if mounted in deployedPage
    deployed_clean = deployed.strip('/')
    matching_deploy = [m for m in mounted_in if deployed_clean in m or (deployed_clean == '' and 'app/page.tsx' in m)]

    results.append({
        'code': code,
        'name': name,
        'deployed': deployed,
        'comp': comp,
        'mounted_count': len(mounted_in),
        'mounted_in': mounted_in,
        'on_target_page': len(matching_deploy) > 0
    })

print(f"{'Code':<5} | {'Status':<10} | {'Target Page':<18} | {'Mounted In Files':<45} | {'Chart Name'}")
print("-" * 120)
for r in results:
    status = "OK" if r['on_target_page'] else ("OTHER" if r['mounted_count'] > 0 else "MISSING!")
    mounted_str = ", ".join([os.path.basename(p) for p in r['mounted_in']]) if r['mounted_in'] else "NONE"
    print(f"{r['code']:<5} | {status:<10} | {r['deployed']:<18} | {mounted_str:<45} | {r['name']}")
