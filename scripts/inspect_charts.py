with open('web/src/app/charts/page.tsx', 'r', encoding='utf-8') as f:
    lines = f.readlines()

code, name, page = None, None, None
for line in lines:
    line = line.strip()
    if line.startswith('code:'):
        code = line.split(':')[1].strip().strip('",')
    elif line.startswith('name:'):
        name = line.split(':', 1)[1].strip().strip('",')
    elif line.startswith('deployedPage:'):
        page = line.split(':', 1)[1].strip().strip('",')
        print(f"{code:5s} | {page:22s} | {name}")
