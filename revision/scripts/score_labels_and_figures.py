# === 14. Score hand labels + audit, and build paper figures (runs standalone after labelling) ===
import os, math, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

OUT = globals().get('OUT', '/content/drive/MyDrive/eurosat_revision')
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k/n:.1%} (95% CI {lo:.1%}-{hi:.1%})" if n else f"{k}/0"

def clean(x):
    x = str(x).strip()
    for c in CLASSES + ['Unclear']:
        if x.lower().replace(' ', '') == c.lower(): return c
    return np.nan if x in ('', 'nan') else 'INVALID:' + x

tiles = pd.read_csv(f'{OUT}/tiles_csplus.csv')

# ---------------- Pack A: single-date accuracy ----------------
A = pd.read_csv(f'{OUT}/packA_labels.csv').merge(pd.read_csv(f'{OUT}/packA_key.csv'), on='label_id')
A = A.merge(tiles[['row', 'col', 'cls_2018', 'cls_2024', 'wc_tree']], on=['row', 'col'])
for yr in ['2018', '2024']:
    A[f'label_{yr}'] = A[f'label_{yr}'].map(clean)
bad = A[[f'label_{y}' for y in ['2018', '2024']]].stack()
bad = bad[bad.astype(str).str.startswith('INVALID')]
if len(bad): print('FIX THESE LABELS (must be a EuroSAT class name or Unclear):', sorted(set(bad))), print()

print('=' * 70, '\nSINGLE-DATE ACCURACY (Pack A, hand labels)\n' + '=' * 70)
for yr in ['2018', '2024']:
    a = A[A[f'label_{yr}'].isin(CLASSES)]
    k = int((a[f'label_{yr}'] == a[f'cls_{yr}']).sum())
    hf, mf = a[f'label_{yr}'] == 'Forest', a[f'cls_{yr}'] == 'Forest'
    print(f"\n{yr}: labelled {len(a)} of {len(A)} tiles ({(A[f'label_{yr}'] == 'Unclear').sum()} Unclear)")
    print(f"  10-class accuracy: {pct(k, len(a))}")
    print(f"  Hand-labelled Forest tiles: {int(hf.sum())}; classifier called Forest: {pct(int((hf & mf).sum()), int(hf.sum()))}  <- forest recall")
    print(f"  Classifier Forest calls that were Forest: {pct(int((hf & mf).sum()), int(mf.sum()))}  <- forest precision")
    conf = a[a[f'label_{yr}'] != a[f'cls_{yr}']].groupby([f'label_{yr}', f'cls_{yr}']).size().sort_values(ascending=False).head(3)
    print('  Top confusions (true -> predicted):')
    for (t, p), n in conf.items(): print(f'    {t} -> {p}: {n}')
    if yr == '2018':
        wcf = a.wc_tree >= 0.5
        print(f"  WorldCover proxy check: WorldCover-forest agrees with hand Forest/non-Forest on {pct(int((wcf == hf).sum()), len(a))}")

notes = A['notes'].fillna('').str.lower()
leaf = notes.str.contains('leaf')
hf18 = A.label_2018 == 'Forest'
print(f"\nLeaf-off: {int((leaf & hf18).sum())} of {int(hf18.sum())} hand-labelled 2018 Forest tiles noted as leafless/brown;")
print(f"  of those, classifier called HerbaceousVegetation: {int((leaf & hf18 & (A.cls_2018 == 'HerbaceousVegetation')).sum())}")
print(f"  of non-leafless Forest tiles, classifier called Forest: {pct(int((~leaf & hf18 & (A.cls_2018 == 'Forest')).sum()), int((~leaf & hf18).sum()))}")
print(f"  of leafless Forest tiles, classifier called Forest:     {pct(int((leaf & hf18 & (A.cls_2018 == 'Forest')).sum()), int((leaf & hf18).sum()))}")

# ---------------- Pack B: blinded audit ----------------
print('\n' + '=' * 70, '\nBLINDED AUDIT (Pack B)\n' + '=' * 70)
B = pd.read_csv(f'{OUT}/packB_audit.csv').merge(pd.read_csv(f'{OUT}/audit_key.csv'), on='audit_id')
B['likely_cause'] = B['likely_cause'].fillna('(blank)').str.strip().str.lower()
B['vis'] = B['real_clearing_visible_YN'].fillna('').str.strip().str.upper().str[:1]
order = ['TP', 'FP', 'FN', 'TN']
print('\nLikely cause by category:'); print(pd.crosstab(B.cat, B.likely_cause).reindex(order).fillna(0).astype(int))
print('\nVisible clearing in 2024 (Y/N) by category:'); print(pd.crosstab(B.cat, B.vis).reindex(order).fillna(0).astype(int))
for yr in ['2018', '2024']:
    col = f'model_{yr}_correct_YN'
    if col in B:
        ok = B[col].fillna('').str.strip().str.upper().str[:1]
        print(f'\nClassifier label correct in {yr}, by category:')
        for c in order:
            s = ok[B.cat == c]; s = s[s.isin(['Y', 'N'])]
            print(f'  {c}: {pct(int((s == "Y").sum()), len(s))}')
if 'clearing_pct_est' in B:
    est = pd.to_numeric(B['clearing_pct_est'], errors='coerce')
    print('\nMedian estimated clearing (% of tile) where a clearing was visible:')
    print(est[B.vis == 'Y'].groupby(B.cat[B.vis == 'Y']).median().reindex(order))

# ---------------- Figure: audit examples (one per category) ----------------
picks = []
for c in order:
    g = B[B.cat == c]
    if not len(g): continue
    top = g.likely_cause.value_counts().index[0]
    picks.append((c, top, g[g.likely_cause == top].iloc[0].audit_id))
fig, axes = plt.subplots(len(picks), 1, figsize=(10, 4.2 * len(picks)))
axes = np.atleast_1d(axes)
names = {'TP': 'True positive', 'FP': 'False positive', 'FN': 'False negative', 'TN': 'True negative'}
for ax, (c, cause, aid) in zip(axes, picks):
    ax.imshow(Image.open(f'{OUT}/packB/{aid}.png')); ax.axis('off')
    ax.set_title(f'({"abcd"[order.index(c)]}) {names[c]} — most common audit cause: {cause.replace("_", " ")}', loc='left', fontsize=11)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_audit_examples.png', dpi=200, bbox_inches='tight'); plt.close()
print('\nSaved fig_audit_examples.png using', [p[2] for p in picks], '(left = 2018, right = 2024; cyan box = tile; orange = 2018 loss, red = 2019-24 loss)')

# ---------------- Figure: synthetic clearing on a log axis ----------------
syn = pd.read_csv(f'{OUT}/synthetic_clearing.csv')
s = syn[syn.frac > 0].groupby(['mode', 'frac']).flipped.agg(['mean', 'size', 'sum']).reset_index()
s['lo'], s['hi'] = zip(*[wilson(int(k), int(n)) for k, n in zip(s['sum'], s['size'])])
fig, ax = plt.subplots(figsize=(7, 4.5))
for mode, lab, mk in [('block', 'One contiguous clearing', 'o'), ('scattered', 'Scattered 40 m clearings', 's')]:
    g = s[s['mode'] == mode]
    ax.plot(g.frac * 100, g['mean'], marker=mk, label=lab)
    ax.fill_between(g.frac * 100, g.lo, g.hi, alpha=0.2)
lf = tiles.filter(regex=r'^loss113_20(19|2[0-4])$').sum(axis=1); lf = lf[lf > 0]
q1, med, q3 = np.percentile(lf * 100, [25, 50, 75])
ax.axvspan(q1, q3, color='grey', alpha=0.15, lw=0, label='Real loss tiles, middle 50%')
ax.axvline(med, color='grey', ls='--', lw=1)
ax.set_xscale('log'); ax.set_xlim(min(0.1, q1 * 0.8), 110)
ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'])
ax.set_xlabel('Cleared area inserted (% of tile, log scale)'); ax.set_ylabel('Share of tiles no longer predicted Forest')
ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(loc='upper left', fontsize=9, frameon=False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_synthetic_clearing.png', dpi=300); plt.close()
print(f'Saved fig_synthetic_clearing.png (real loss tiles: median {med:.2f}%, IQR {q1:.2f}-{q3:.2f}%)')
