#!/usr/bin/env python3
"""fetch_ct_crop.py - local SPARSE copy of the PHerc 343 8.64 um CT (level 0) holding only the 128^3 chunks a render crop of a
tifxyz mesh can touch: every mesh node in the window (render px x0..x1, y0..y1; one node = 20 px) +- MARGIN voxels in z, y and x
(a 66-layer render reaches +-33 voxels along the normal). Chunks that are absent on the server read as the fill value (0).
Writes .zgroup/.zattrs and 0/.zarray verbatim from the open-data bucket, the .zarray of the other pyramid levels (metadata only,
so renderers that open every listed level work), the level-0 chunk files, and chunks.sha256 (one line per chunk).
  python3 fetch_ct_crop.py --mesh concat_w047-w048_R5B2_z9500-11000.tifxyz --out PHerc0343_L0_crop.zarr [--window 2560 320 5760 1600]
          [--margin 48] [--dry-run]
For the crop in README.md: 716 chunks, 1.50 GB. MIT licence."""
import argparse, hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, tifffile
BASE = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr'
ap = argparse.ArgumentParser(); ap.add_argument('--mesh', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--window', type=int, nargs=4, default=[2560, 320, 5760, 1600]); ap.add_argument('--margin', type=int, default=48)
ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
X, Y, Z = [tifffile.imread(os.path.join(a.mesh, f'{c}.tif')) for c in 'xyz']
x0, y0, x1, y1 = a.window
sl = (slice(max(0, y0 // 20 - 1), y1 // 20 + 2), slice(max(0, x0 // 20 - 1), x1 // 20 + 2))
ok = X[sl] > 0
P = np.stack([Z[sl][ok], Y[sl][ok], X[sl][ok]], 1)
m = a.margin; ch = set()
for dz in (-m, 0, m):
    for dy in (-m, 0, m):
        for dx in (-m, 0, m):
            for q in ((P + np.array([dz, dy, dx])) // 128).astype(int):
                ch.add(tuple(int(v) for v in q))
ch = sorted(ch)
print(f'{len(ch)} chunks, {len(ch) * 128 ** 3 / 1e9:.2f} GB (uncompressed u8, compressor null)')
if a.dry_run:
    sys.exit(0)
def get(u):
    for t in range(6):
        try:
            with urllib.request.urlopen(u, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
        except Exception:
            pass
    raise RuntimeError('failed ' + u)
os.makedirs(os.path.join(a.out, '0'), exist_ok=True)
for meta in ('.zgroup', '.zattrs'):
    b = get(f'{BASE}/{meta}')
    if b is not None:
        open(os.path.join(a.out, meta), 'wb').write(b)
open(os.path.join(a.out, '0', '.zarray'), 'wb').write(get(f'{BASE}/0/.zarray'))
for lv in range(1, 8):                                   # other pyramid levels: metadata only (no chunks), so readers
    zb = get(f'{BASE}/{lv}/.zarray')                     # that open the multiscale group find every level it lists
    if zb is None:
        break
    os.makedirs(os.path.join(a.out, str(lv)), exist_ok=True); open(os.path.join(a.out, str(lv), '.zarray'), 'wb').write(zb)
def one(c):
    z, y, x = c; f = os.path.join(a.out, '0', str(z), str(y), str(x))
    if os.path.exists(f):
        return c, hashlib.sha256(open(f, 'rb').read()).hexdigest()
    b = get(f'{BASE}/0/{z}/{y}/{x}')
    if b is None:
        return c, 'absent'
    os.makedirs(os.path.dirname(f), exist_ok=True); open(f + '.part', 'wb').write(b); os.replace(f + '.part', f)
    return c, hashlib.sha256(b).hexdigest()
with ThreadPoolExecutor(32) as ex:
    res = list(ex.map(one, ch))
with open(os.path.join(a.out, 'chunks.sha256'), 'w') as f:
    for (z, y, x), h in res:
        f.write(f'{h}  0/{z}/{y}/{x}\n')
print('done;', sum(h == 'absent' for _, h in res), 'chunks absent on S3 (fill 0)')
