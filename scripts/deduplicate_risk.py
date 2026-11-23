import re

with open('web/src/app/risk/RiskClientView.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern for the entire Row 6 block
row6_pattern = r'\s*\{\/\* Row 6: Non-Linear Joint Tail Density \(C18\) & Ridgeline Return Distributions \(C19\) \*\/\}[\s\S]*?<RidgelineDensityChart \/>\s*<\/ChartCard>\s*<\/div>\s*<\/div>'

matches = list(re.finditer(row6_pattern, content))
print(f"Found {len(matches)} occurrences of Row 6!")

if len(matches) > 0:
    # Save one clean copy
    clean_row6 = matches[0].group(0).strip()
    
    # Remove all occurrences
    content_clean = re.sub(row6_pattern, '', content)
    
    # Find where {/* Drawers */} starts and insert right before it
    drawer_idx = content_clean.find('{/* Drawers */}')
    if drawer_idx != -1:
        content_final = content_clean[:drawer_idx] + "\n      " + clean_row6 + "\n\n      " + content_clean[drawer_idx:]
        with open('web/src/app/risk/RiskClientView.tsx', 'w', encoding='utf-8') as f:
            f.write(content_final)
        print("Successfully removed all duplicates and placed Row 6 exactly once before Drawers!")
    else:
        print("Error: Could not find {/* Drawers */} in file")
else:
    print("No matches found for Row 6 pattern")
