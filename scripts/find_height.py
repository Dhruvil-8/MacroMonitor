import glob

for f in glob.glob('web/src/**/*.tsx', recursive=True):
    with open(f, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    for i, l in enumerate(lines):
        if 'height:' in l and '100%' in l:
            print(f"{f}:{i+1} -> {l.strip()}")
