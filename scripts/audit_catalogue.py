with open('web/src/app/charts/page.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Split by catalogue items
items = re.findall(r'\{\s*id:\s*"(c\d+)",\s*code:\s*"(C\d+)",\s*name:\s*"([^"]+)",\s*category:\s*"([^"]+)",\s*idea:\s*"([^"]+)",\s*insight:\s*"([^"]+)",\s*deployedPage:\s*"([^"]+)",\s*renderComponent:\s*\(\)\s*=>\s*([\s\S]*?)\n\s*\},', content)

print(f"Parsed {len(items)} items")
for item in items:
    cid, code, name, cat, idea, insight, page, comp = item
    is_dummy = "Math.random" in comp or "[id]" in page
    comp_name = comp.strip().split('\n')[0][:50]
    print(f"{code:4s} | Page: {page:16s} | Dummy/Flag: {str(is_dummy):5s} | Comp: {comp_name}")
