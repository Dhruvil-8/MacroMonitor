import yaml

with open('pipeline/config/params.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
datasets = cfg.get('datasets', [])
print(f"Total datasets in params.yaml: {len(datasets)}")
mapping = {d.get('id'): d.get('source_symbol') for d in datasets}
print(list(mapping.items())[:20])
