import csv, json

with open('audit/results/ui_value_equality_results.csv', 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        print(f"{row['route']}: Status={row['status']}, Charts={row['charts_found']}, Requests={row['requests_count']}")
