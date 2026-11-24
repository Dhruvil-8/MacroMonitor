import csv

with open('audit/findings.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

non_future = [r for r in rows if not r['id'].startswith('DATA_FUTURE_DATES')]
print(f"Total findings: {len(rows)}, Non-future findings: {len(non_future)}")
for r in non_future:
    print(f"[{r['severity']}] {r['id']}: {r['location']} -> {r['actual']}")
