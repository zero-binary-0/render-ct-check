#!/usr/bin/env python3
"""render_ct_check.py - check that a rendered surface volume agrees with the CT it was rendered from.

For every pixel of the render's surface layer, the "truth" is the CT sampled (trilinear) at the mesh point that pixel
belongs to: the mean of two samples at -0.25 and +0.25 voxel along the mesh normal. The render is split into 256 x 256 px
blocks; for each block the Pearson r between render and truth is computed, plus a pooled r over all valid pixels.

PASS iff  pooled r >= 0.85  AND  >= 95 % of valid blocks have r >= 0.80  AND  no 3 x 3 group of valid blocks is entirely < 0.80.
A block is valid if >= 50 % of its pixels have a finite truth (mesh valid, inside the CT) and are not masked by --fold.

Optional --shift-test: the truth is moved by -2, -1, -0.5, +0.5, +1, +2 voxels along the normal. A check that can see
1-voxel surface errors must FAIL at +-1 and +-2; +-0.5 is reported only.

Inputs: a tifxyz mesh (x.tif, y.tif, z.tif, meta.json with "scale"), the render (a folder of 00.tif, 01.tif, ... layers or a
zarr whose array "0" (or the root array) has shape (layers, H, W)) and the CT as a zarr (array "0" or the root array, z/y/x).
Render pixel (row, col) maps to mesh grid position ((row + Y0) * scale, (col + X0) * scale); --origin X0 Y0 if the render is
a crop of the full render. Surface layer: centre layer for an odd layer count, the two centre layers (block r averaged)
for an even count, or --layer K.

  python render_ct_check.py --mesh SEG.tifxyz --render render_dir_or.zarr --ct volume.zarr [--origin X0 Y0] [--fold mask.npy]
                            [--layer K] [--shift-test] [--out result.json]
MIT licence. Dependencies: numpy, tifffile, zarr."""
import argparse, glob, json, os, sys
import numpy as np
import tifffile
import zarr

BLOCK, THR_BLOCK, THR_SHARE, THR_POOLED, MIN_VALID_FRAC = 256, 0.80, 0.95, 0.85, 0.5
SHIFTS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def open_array(path):
    z = zarr.open(path, mode='r')
    return z['0'] if hasattr(z, 'keys') and '0' in z else z


class Render:
    def __init__(self, path):
        if os.path.isdir(path) and glob.glob(os.path.join(path, '[0-9][0-9].tif')):
            self.files = sorted(glob.glob(os.path.join(path, '[0-9][0-9].tif'))); self.n = len(self.files); self.arr = None
        else:
            self.arr = open_array(path); self.n = self.arr.shape[0]

    def layer(self, k):
        return tifffile.imread(self.files[k]) if self.arr is None else np.asarray(self.arr[k])


class Mesh:
    def __init__(self, path):
        meta = json.load(open(os.path.join(path, 'meta.json')))
        sc = meta['scale'][0] if isinstance(meta['scale'], list) else meta['scale']
        self.step = 1.0 / sc                                   # render px per mesh cell
        M = [tifffile.imread(os.path.join(path, c + '.tif')).astype(np.float64) for c in 'xyz']
        bad = np.zeros(M[0].shape, bool)
        for m in M:
            bad |= (m <= -1) | ~np.isfinite(m)
        self.G = np.stack([np.where(bad, np.nan, m) for m in M])           # (3, rows, cols) xyz
        dr = np.full_like(self.G, np.nan); dc = np.full_like(self.G, np.nan)
        dr[:, 1:-1, :] = self.G[:, 2:, :] - self.G[:, :-2, :]; dc[:, :, 1:-1] = self.G[:, :, 2:] - self.G[:, :, :-2]
        n = np.cross(dc, dr, axis=0); self.N = n / np.sqrt((n ** 2).sum(0))   # unit normal at the nodes

    def interp(self, F, row0, row1, col0, col1, x0, y0):
        """Bilinear interpolation of a node field F (3, rows, cols) at render pixels rows row0..row1, cols col0..col1."""
        gr = (y0 + np.arange(row0, row1, dtype=np.float64)) / self.step; gc = (x0 + np.arange(col0, col1, dtype=np.float64)) / self.step
        i0 = np.floor(gr).astype(int); j0 = np.floor(gc).astype(int); fr = (gr - i0)[:, None]; fc = (gc - j0)[None, :]
        okr = (i0 >= 0) & (i0 + 1 < F.shape[1]); okc = (j0 >= 0) & (j0 + 1 < F.shape[2])
        i0 = np.clip(i0, 0, F.shape[1] - 2); j0 = np.clip(j0, 0, F.shape[2] - 2)
        out = np.empty((3, row1 - row0, col1 - col0))
        for k in range(3):
            g = F[k]
            out[k] = (g[i0][:, j0] * (1 - fr) * (1 - fc) + g[i0][:, j0 + 1] * (1 - fr) * fc + g[i0 + 1][:, j0] * fr * (1 - fc) + g[i0 + 1][:, j0 + 1] * fr * fc)
        out[:, ~okr, :] = np.nan; out[:, :, ~okc] = np.nan
        return out


def sample(ct, P):
    """Trilinear CT sample at points P = (x, y, z) arrays (voxel units); NaN outside the volume or where P is NaN."""
    x, y, z = P; v = np.isfinite(x) & np.isfinite(y) & np.isfinite(z); out = np.full(x.shape, np.nan)
    if not v.any():
        return out
    xv, yv, zv = x[v], y[v], z[v]; x0, y0, z0 = np.floor(xv).astype(int), np.floor(yv).astype(int), np.floor(zv).astype(int)
    lo = [max(int(a.min()), 0) for a in (z0, y0, x0)]; hi = [min(int(a.max()) + 2, s) for a, s in zip((z0, y0, x0), ct.shape)]
    if any(h <= l for h, l in zip(hi, lo)):
        return out
    sub = np.asarray(ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float64)
    ok = (z0 >= lo[0]) & (y0 >= lo[1]) & (x0 >= lo[2]) & (z0 + 1 < hi[0]) & (y0 + 1 < hi[1]) & (x0 + 1 < hi[2])
    fx, fy, fz = (xv - x0)[ok], (yv - y0)[ok], (zv - z0)[ok]; iz, iy, ix = (z0 - lo[0])[ok], (y0 - lo[1])[ok], (x0 - lo[2])[ok]
    s = np.zeros(fx.shape)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                s += (fz if dz else 1 - fz) * (fy if dy else 1 - fy) * (fx if dx else 1 - fx) * sub[iz + dz, iy + dy, ix + dx]
    out.ravel()[np.flatnonzero(v)[ok]] = s
    return out


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.size > 10 and a.std() > 0 and b.std() > 0 else float('nan')


def evaluate(mesh, ct, layers, fold, x0, y0, shift=0.0):
    H, W = layers[0].shape; nbr, nbc = H // BLOCK, W // BLOCK
    R = np.full((nbr, nbc), np.nan); VAL = np.zeros((nbr, nbc), bool); acc = [[0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in layers]
    for bi in range(nbr):
        for bj in range(nbc):
            row0, col0 = bi * BLOCK, bj * BLOCK
            P = mesh.interp(mesh.G, row0, row0 + BLOCK, col0, col0 + BLOCK, x0, y0)
            N = mesh.interp(mesh.N, row0, row0 + BLOCK, col0, col0 + BLOCK, x0, y0); N = N / np.sqrt((N ** 2).sum(0))
            Ps = P + shift * N
            T = 0.5 * (sample(ct, Ps - 0.25 * N) + sample(ct, Ps + 0.25 * N)); m = np.isfinite(T)
            if fold is not None:
                m &= ~fold[row0:row0 + BLOCK, col0:col0 + BLOCK]
            if m.mean() < MIN_VALID_FRAC:
                continue
            VAL[bi, bj] = True; t = T[m]; rs = []
            for k, lay in enumerate(layers):
                q = lay[row0:row0 + BLOCK, col0:col0 + BLOCK][m].astype(np.float64); rs.append(corr(t, q)); a = acc[k]
                a[0] += t.size; a[1] += t.sum(); a[2] += q.sum(); a[3] += (t * t).sum(); a[4] += (q * q).sum(); a[5] += (t * q).sum()
            R[bi, bj] = np.nanmean(rs) if not np.isnan(rs).all() else np.nan
    def pooled(a):
        n = a[0]
        if not n:
            return float('nan')
        cov = a[5] / n - a[1] / n * a[2] / n; va = a[3] / n - (a[1] / n) ** 2; vb = a[4] / n - (a[2] / n) ** 2
        return float(cov / np.sqrt(va * vb))
    pr = float(np.mean([pooled(a) for a in acc])); nv = int(VAL.sum())
    bad = VAL & ~(R >= THR_BLOCK); share = float((R[VAL] >= THR_BLOCK).sum() / nv) if nv else float('nan')
    cluster = [[int(i), int(j)] for i in range(nbr - 2) for j in range(nbc - 2) if bad[i:i + 3, j:j + 3].all()]
    ok = bool(nv > 0 and pr >= THR_POOLED and share >= THR_SHARE and not cluster)
    return {'shift_vox': shift, 'blocks_total': int(nbr * nbc), 'blocks_valid': nv, 'blocks_failing': int(bad.sum()),
            'share_blocks_ge_0_80': share, 'pooled_r': pr, 'cluster_3x3_all_failing': cluster[:20], 'verdict': 'PASS' if ok else 'FAIL',
            'failing_blocks': [{'block_row': int(i), 'block_col': int(j), 'px_row0': int(i * BLOCK), 'px_col0': int(j * BLOCK),
                                'r': (None if np.isnan(R[i, j]) else round(float(R[i, j]), 4))} for i, j in zip(*np.where(bad))],
            'too_few_blocks_warning': nv < 36}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh', required=True, help='tifxyz folder (x.tif, y.tif, z.tif, meta.json)')
    ap.add_argument('--render', required=True, help='folder of NN.tif layers, or a zarr (layers, H, W)')
    ap.add_argument('--ct', required=True, help='CT volume zarr (z, y, x); array "0" or the root array')
    ap.add_argument('--origin', type=int, nargs=2, default=(0, 0), metavar=('X0', 'Y0'), help='render-pixel origin if the render is a crop')
    ap.add_argument('--fold', help='optional bool .npy mask (render shape); True pixels are ignored')
    ap.add_argument('--layer', type=int, help='surface layer index (default: centre layer(s))')
    ap.add_argument('--shift-test', action='store_true', help='also run the truth shifted by +-0.5, +-1, +-2 voxels')
    ap.add_argument('--out', help='write the full result as JSON')
    a = ap.parse_args(argv)
    mesh, ren, ct = Mesh(a.mesh), Render(a.render), open_array(a.ct)
    gate = [a.layer] if a.layer is not None else ([(ren.n - 1) // 2] if ren.n % 2 else [ren.n // 2 - 1, ren.n // 2])
    layers = [ren.layer(k) for k in gate]
    fold = np.load(a.fold).astype(bool) if a.fold else None
    if fold is not None and fold.shape != layers[0].shape:
        sys.exit(f'--fold shape {fold.shape} != render shape {layers[0].shape}')
    x0, y0 = a.origin
    res = {'tool': 'render_ct_check.py', 'render_layers': ren.n, 'surface_layers': gate, 'render_shape': list(layers[0].shape),
           'rule': f'PASS iff pooled r >= {THR_POOLED} and >= {THR_SHARE:.0%} of valid {BLOCK}-px blocks have r >= {THR_BLOCK} and no 3x3 group of valid blocks all < {THR_BLOCK}',
           'truth': 'CT trilinear, mean of samples at -0.25 and +0.25 voxel along the mesh normal', **evaluate(mesh, ct, layers, fold, x0, y0)}
    print(f"render {a.render}: surface layer(s) {gate} of {ren.n}, {res['blocks_valid']} valid blocks of {res['blocks_total']}")
    print(f"pooled r {res['pooled_r']:.3f} | blocks with r >= {THR_BLOCK}: {res['share_blocks_ge_0_80']:.1%} | failing blocks {res['blocks_failing']}"
          f" | 3x3 failing clusters {len(res['cluster_3x3_all_failing'])}")
    if res['too_few_blocks_warning']:
        print('WARNING: fewer than 36 valid blocks: the 95 % share rule is coarse at this size')
    print(f"{res['verdict']}  ({res['rule']})")
    if a.shift_test:
        res['shift_test'] = [evaluate(mesh, ct, layers, fold, x0, y0, s) for s in SHIFTS]
        print('shift test (truth moved along the normal; +-1 and +-2 should FAIL, +-0.5 is information):')
        for s in res['shift_test']:
            print(f"  {s['shift_vox']:+.1f} vox: pooled r {s['pooled_r']:.3f}, failing {s['blocks_failing']}/{s['blocks_valid']} -> {s['verdict']}")
        res['shift_test_sensitive'] = all(s['verdict'] == 'FAIL' for s in res['shift_test'] if abs(s['shift_vox']) >= 1)
        if not res['shift_test_sensitive']:
            print('WARNING: a 1- or 2-voxel shift still passes: on this data the check cannot detect 1-voxel surface errors')
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    return res


if __name__ == '__main__':
    main()
