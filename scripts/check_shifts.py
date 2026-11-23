import json

with open('web/public/data/v1/divergences.json') as f:
    d = json.load(f)

for p in d['data']['pairs']:
    dates = p['cusum']['dates']
    dates_set = set(dates)
    shifts = p['cusum']['shifts']
    in_win = [s for s in shifts if s['date'] in dates_set]
    out_win = [s for s in shifts if s['date'] not in dates_set]
    min_date = dates[0] if dates else 'N/A'
    max_date = dates[-1] if dates else 'N/A'
    print(f"{p['id']}: dates range [{min_date} to {max_date}] (len {len(dates)})")
    print(f"   total shifts: {len(shifts)}, in_window: {len(in_win)}, out_of_window: {len(out_win)}")
    if out_win:
        print(f"   out dates: {[s['date'] for s in out_win]}")
