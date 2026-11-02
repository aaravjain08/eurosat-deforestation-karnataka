# === Figure 2: study region, with "tree-cover loss" wording ===
# Needs Drive mounted. Uses files already in eurosat_revision/ from the revision notebook.
import json, urllib.request
import numpy as np, rasterio, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import maximum_filter

OUT = '/content/drive/MyDrive/eurosat_revision'

# Sentinel-2 Jan-Mar 2018 composite (bands: B4 B3 B2 B8 nobs) and reference stack
with rasterio.open(f'{OUT}/comp_csplus_2018.tif') as src:
    comp = src.read([1, 2, 3]).astype(np.float32)
    b = src.bounds
with rasterio.open(f'{OUT}/reference.tif') as src:
    lossyear = src.read(1)      # lossyear_v111
    tc2000 = src.read(3)        # treecover2000

# true colour, 2-98 % stretch per band
rgb = np.zeros(comp.shape[1:] + (3,), np.float32)
for i in range(3):
    band = comp[i]; v = band[band > 0]
    lo, hi = np.percentile(v, [2, 98])
    rgb[..., i] = np.clip((band - lo) / (hi - lo), 0, 1)

# Hansen v1.11 tree-cover loss 2018-2023, >=30 % canopy in 2000
loss = np.isin(lossyear, range(18, 24)) & (tc2000 >= 30)
print(f'Loss pixels: {loss.sum()}  (~{loss.sum() * 9.66 * 10 / 1e4:.1f} ha)')
big = maximum_filter(loss, size=9)          # enlarged for visibility only
overlay = rgb.copy(); overlay[big] = [0.85, 0.05, 0.05]

# India outline (Natural Earth, 1:110m)
url = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson'
world = json.load(urllib.request.urlopen(url))
india = next(f for f in world['features'] if f['properties'].get('ADMIN') == 'India')
polys = india['geometry']['coordinates']
if india['geometry']['type'] == 'Polygon': polys = [polys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.4), gridspec_kw={'width_ratios': [1, 1.15]})
for poly in polys:
    xy = np.array(poly[0]); ax1.fill(xy[:, 0], xy[:, 1], color='#e6e6e6', ec='#555', lw=0.6)
cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
ax1.plot(cx, cy, 'o', ms=9, mfc='none', mec='red', mew=1.6, label='Study region')
ax1.set_xlim(66, 98); ax1.set_ylim(6, 37); ax1.set_aspect('equal')
ax1.set_xlabel('Longitude (°E)'); ax1.set_ylabel('Latitude (°N)')
ax1.legend(loc='lower left', fontsize=8, frameon=False)
ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes, va='top', fontweight='bold')

ax2.imshow(overlay, extent=[b.left, b.right, b.bottom, b.top], interpolation='nearest')
ax2.set_xlabel('Longitude (°E)'); ax2.set_ylabel('Latitude (°N)')
ax2.legend(handles=[Patch(color=(0.85, 0.05, 0.05), label='Tree-cover loss, 2018–2023')],
           loc='upper right', fontsize=8, framealpha=0.85)
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes, va='top', fontweight='bold', color='white')
# 10 km scale bar
deg10km = 10 / (111.32 * np.cos(np.radians(cy)))
x0, y0 = b.left + 0.04 * (b.right - b.left), b.bottom + 0.05 * (b.top - b.bottom)
ax2.plot([x0, x0 + deg10km], [y0, y0], color='white', lw=3)
ax2.text(x0 + deg10km / 2, y0 + 0.012, '10 km', color='white', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT}/figure2_study_region (1).pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig2_study_region.png', dpi=250, bbox_inches='tight')
plt.show()
print('Saved figure2_study_region (1).pdf and fig2_study_region.png')
