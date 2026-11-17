#!/usr/bin/env python3
"""Synthetic test for render_ct_check.py (no download, well under a minute).
CT = independent smoothed-noise slices (so a 1-voxel depth error decorrelates), a flat tifxyz mesh at z = 6, renders built
by sampling the CT the way a renderer would. Checks: a correct render PASSES; a render one voxel off the surface FAILS; a
render with one wrong strip FAILS; the shift self-test fails at +-1 and +-2 voxels; the zarr render input works.
Works with zarr 2.18 and zarr 3.x.
  python tests/test_render_ct_check.py"""
import json, os, sys, tempfile
import numpy as np, tifffile, zarr
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import render_ct_check as rcc


def write_array(path, data, chunks):
    """A group with one array '0' (works with zarr 2.x and zarr 3.x)."""
    zarr.open_group(path, mode='w')
    a = zarr.open_array(os.path.join(path, '0'), mode='w', shape=data.shape, chunks=chunks, dtype=data.dtype)
    a[...] = data


def build(tmp):
    rng = np.random.default_rng(0); S, ZC, CTN = 1024, 6, 1100
    ct = np.stack([gaussian_filter(rng.standard_normal((CTN, CTN)), 2.0) for _ in range(12)])
    ct = np.clip(128 + ct / ct.std() * 40, 0, 255).astype(np.uint8)
    write_array(os.path.join(tmp, 'ct.zarr'), ct, (12, 256, 256))
    m = os.path.join(tmp, 'mesh.tifxyz'); os.makedirs(m); n = 54; ii, jj = np.mgrid[0:n, 0:n].astype(np.float32)
    for c, v in zip('xyz', (jj * 20, ii * 20, np.full_like(ii, ZC))):
        tifffile.imwrite(os.path.join(m, c + '.tif'), v)
    json.dump({'type': 'seg', 'format': 'tifxyz', 'scale': [0.05, 0.05]}, open(os.path.join(m, 'meta.json'), 'w'))
    def render(name, surface):
        d = os.path.join(tmp, name); os.makedirs(d)
        for k, z in enumerate((ZC - 1, ZC, ZC + 1)):
            img = surface if k == 1 else ct[z, :S, :S]
            tifffile.imwrite(os.path.join(d, f'{k:02d}.tif'), np.asarray(img, np.float32))
        return d
    good = ct[ZC, :S, :S].astype(np.float32) + rng.standard_normal((S, S)).astype(np.float32) * 6
    off1 = ct[ZC + 1, :S, :S].astype(np.float32)
    strip = good.copy(); strip[:, 512:768] = ct[ZC + 3, :S, 512:768]
    write_array(os.path.join(tmp, 'good.zarr'), np.stack([ct[ZC - 1, :S, :S], good, ct[ZC + 1, :S, :S]]).astype('f4'), (1, 512, 512))
    return m, os.path.join(tmp, 'ct.zarr'), render('good', good), render('off1', off1), render('strip', strip), os.path.join(tmp, 'good.zarr')


def main():
    C = {}
    with tempfile.TemporaryDirectory() as tmp:
        mesh, ct, good, off1, strip, goodz = build(tmp)
        q = lambda r, *x: rcc.main(['--mesh', mesh, '--render', r, '--ct', ct, *x])
        r = q(good, '--shift-test'); C['correct render PASSES'] = r['verdict'] == 'PASS'
        C['shift test: +-1 and +-2 voxel FAIL'] = r['shift_test_sensitive']
        C['shift test: +-0.5 reported'] = any(abs(s['shift_vox']) == 0.5 for s in r['shift_test'])
        r = q(off1); C['render one voxel off the surface FAILS'] = r['verdict'] == 'FAIL' and r['blocks_failing'] == r['blocks_valid']
        r = q(strip); C['render with one wrong strip FAILS (only that strip)'] = r['verdict'] == 'FAIL' and {b['block_col'] for b in r['failing_blocks']} == {2}
        r = q(goodz); C['zarr render input PASSES'] = r['verdict'] == 'PASS'
    for k, v in C.items():
        print(('ok   ' if v else 'FAIL ') + k)
    print(f'{sum(C.values())}/{len(C)} checks passed')
    sys.exit(0 if all(C.values()) else 1)


if __name__ == '__main__':
    main()
