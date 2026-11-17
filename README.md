# render_ct_check: does a rendered surface volume agree with the CT it came from?

`render_ct_check.py` takes a tifxyz mesh, a rendered surface volume (layer TIFFs or a zarr) and the CT volume. For each pixel of the render's surface layer it compares the rendered value with the CT sampled at the mesh point that pixel belongs to. If the render matches the CT at the mesh, the two agree closely. Where they do not, the tool says **where**, block by block, in the `--out` JSON.

With it we found that, on one published segment, the public renderer image `ghcr.io/scrollprize/villa/volume-cartographer:edge`, run with 8 threads, behaved in two ways. On the standard crop window it exits with code 0 and prints no error or warning in any captured run, yet writes surface values that **do not match the CT** in a located part of the render, and the result **changes from run to run**. On a second window of the same segment it **crashed** (exit code 139). With `OMP_NUM_THREADS=1` the same image completes and passes the check on both windows (second window: one render each, 24 blocks), and the current release passes on both (see *Finding*).

## What it computes
- **Truth:** the CT sampled trilinearly at the mesh point of each render pixel: the **mean of two samples at −0.25 and +0.25 voxel along the mesh normal** (see *Why ±0.25*).
  - The mesh grid is upsampled bilinearly. Render pixel (row, col) maps to grid position ((row + Y0) · scale, (col + X0) · scale), with scale from tifxyz `meta.json`.
  - `--origin X0 Y0` is needed when the render is a crop.
- **Surface layer:** the centre layer for an odd layer count, the two centre layers for an even count, or `--layer K`.
- **Block scores:** Pearson r between render and truth for each 256 × 256 px block.
  - For two surface layers the block r is the mean of the two layers' r.
  - A block is valid if at least 50 % of its pixels have a finite truth and are not masked by `--fold`. Partial edge blocks are dropped.
  - **A valid block whose r is undefined (NaN, e.g. no variance) counts as failing** and counts in the share.
- **Pooled r:** over all valid pixels; with two surface layers it is **the mean of the two layers' pooled r**.
- **Fixed rule:** **PASS** iff
  - pooled r ≥ 0.85, AND
  - at least 95 % of valid blocks have r ≥ 0.80, AND
  - no 3 × 3 group of valid blocks lies entirely below 0.80.

  With fewer than 36 valid blocks the share rule is coarse, and the tool warns.
- **`--shift-test`:** re-scores with the truth moved by −2, −1, −0.5, +0.5, +1, +2 voxels along the normal. If ±1 and ±2 do not fail, the tool warns that the data cannot resolve 1-voxel errors. ±0.5 is information.

```
python render_ct_check.py --mesh SEGMENT.tifxyz --render render_dir_or.zarr --ct volume.zarr [--origin X0 Y0] [--fold mask.npy] [--layer K] [--shift-test] [--out result.json]
```
- **Output:** a short summary on stdout (pooled r, share of good blocks, failing blocks, 3 × 3 clusters, PASS/FAIL with the rule), and with `--out` a JSON file with every failing block's position and r.
- **Requirements:** numpy, tifffile, zarr; the test also needs scipy. The test (`tests/test_render_ct_check.py`) was tested 6/6 with:
  - Python 3.12 + zarr 2.18.7;
  - Python 3.14 + zarr 3.4.2.

  Also reported by the author, not independently re-run: Python 3.11.2 + zarr 2.18.7 and Python 3.11.2 + zarr 3.1.6.

## Finding (one segment, one crop, one image digest, one machine)
**Data (all public):**
- the PHerc 343 First Letters winner's segment `concat_w047-w048_R5B2_z9500-11000.tifxyz`;
- the standard render crop of 3200 × 1280 px at x 2560, y 320 (render px), and a second window, its right half (1600 × 1280 px at x 4160, y 320; chosen by a rule fixed before rendering);
- 66 layers, surface layers 32/33, `--group-idx 0`, identical flags for both tools.

**Renderers:**
- the current VC3D release AppImage `e413746`, with `--surface-interpolation bicubic`;
- the container `ghcr.io/scrollprize/villa/volume-cartographer:edge`, digest `sha256:bad516f6…` (built 2026-05-13; on 2026-11-16 `:edge` still resolves to it). It has no interpolation option (its help does not state the method) and no threads option, so threads were set with OMP_NUM_THREADS only.

| render | threads | repeat runs | pooled r | failing blocks of 54 | 3 × 3 failing clusters | verdict |
|---|---|---|---|---|---|---|
| container, run 1 | 8 | — | 0.671 | 25 | 6 | FAIL |
| container, rerun | 8 | differs from run 1 in all 66 layers | 0.585 | 29 | 6 | FAIL |
| container, A | 8 | differs from every other run | 0.609 | 27 | 7 | FAIL |
| container, B | 8 | differs from every other run | 0.576 | 28 | 6 | FAIL |
| container, E (`--cache-gb 16`) | 8 | differs from every other run | 0.582 | 31 | 9 | FAIL |
| container, T2a / T2b | 2 | the two differ in all 66 layers (96.9 % identical pixels in the surface layer) | 0.910 / 0.918 | 7 / 5 | — | FAIL / FAIL |
| container, T4a / T4b | 4 | the two differ in all 66 layers | 0.725 / 0.760 | 27 / 23 | — | FAIL / FAIL |
| container, C and D | **1** | **C and D byte-identical** | **0.953** | **0** | 0 | **PASS** |
| current release `e413746` (three renders) | 8 | **all identical** in all 66 pixel layers | **0.933** | **0** | 0 | **PASS** |

**Second window** (x 4160, y 320, 1600 × 1280 px; 30 blocks, 24 valid, so the tool warns that the share rule is coarse). A pass on 24 valid blocks is weaker evidence than on 54 (one failing block is 4 %), and the one-thread and release renders of this window were single renders, not repeated.

| render | threads | exit code | pooled r | failing of 24 valid | verdict |
|---|---|---|---|---|---|
| container, W8a / W8b | 8 | **139 / 139** (crash; partial output) | not scored (crash, partial output) | - | CRASH (exit 139) |
| container, W1a | 1 | 0 | 0.958 | 0 | PASS |
| current release `e413746` | 8 | 0 | 0.937 | 0 | PASS |

We ran the check on the two partial output folders only because they existed (pooled r 0.124 / 0.129; 24 / 22 failing of the 24 blocks); these numbers describe incomplete output, not the quality of a render.

**Machine:** AMD Ryzen 9 9900X (12 cores, 24 logical CPUs), WSL2 Ubuntu 24.04 on Windows. `nproc` is 24 on the host and inside the container. The image sets no `OMP_NUM_THREADS`, and the tool has no thread option.

**What we observed:**
- **Exit code and output on the standard window:** every container render exited with code 0. Its captured output (8 lines, the same text in every captured run apart from the progress bar and the output path; run 1's own output was not kept) contains no line matching error, warning, fail, exception, abort or invalid. It prints `Voxel size: 1.0 (no metadata found; override with --voxel-size)` and no interpolation line. (The current release prints one warning line, about the volume's voxel size, and `Surface interpolation: bicubic`.)
- **A crash on the second window:** both 8-thread container renders of the second window crashed with exit code 139. They printed `terminate called after throwing an instance of 'cv::Exception'`, and one of them also printed an OpenCV assertion (`matrix.cpp:808`, the check that a region of interest lies inside the matrix). Each left a partial zarr. With `OMP_NUM_THREADS=1` the same window completed with exit 0 and passes.
- **Same region every time:** in **every** 8-thread container render of the standard window, **all 20 blocks** of pixel columns 0-1023 × rows 0-1279 fail (block r from −0.14 to 0.79 in run 1).
- **The rest moves:** the further failing blocks to the right change from run to run (run 1: [1,4] [1,6] [1,11] [2,4] [3,6]; the other runs: 7-11 blocks at other positions).
- **Plausible-looking but wrong:** over raw columns 0-900, run 1's surface layer looks normal (mean 92.1, SD 36.9, no empty pixels). But it correlates with the CT at only r 0.19, against 0.95 for the release. No shift, flip or layer offset explains it. Elsewhere the two renderers agree (r about 0.95).
- **`OMP_NUM_THREADS=1`:** with `OMP_NUM_THREADS=1` the same image renders repeatably (two runs byte-identical) and **passes** (0 of 54 failing).
- **Thread count:** with 2 threads the two renders differ and fail, mildly (5-7 of 54 blocks, all in the left strip, block columns 0-1). With 4 threads: 23-27 failing (15-18 of them in the left strip, the rest further right). With 8 threads: 25-31 failing, all 20 left-strip blocks every time. With 1 thread: repeatable, 0 failing. In these runs (two renders per setting on the standard window) the number of failing blocks and the drop in pooled r rise with the thread count (1 thread: 0 failing; 2: 5-7; 4: 23-27; 8: 25-31), and only OMP_NUM_THREADS=1 passes. This is consistent with a multithreading problem in that build, but **we have not identified the cause, and we did not bisect.**
- **Neither good render is "the" truth:** the single-thread container render and the release render are not pixel-identical (26 % identical pixels in the surface layer, r about 0.977; the old image has no bicubic option). The CT check is the reference, and both pass it.

**Shift test on the current release's render:**

| truth shift (voxel) | −2 | −1 | −0.5 | +0.5 | +1 | +2 |
|---|---|---|---|---|---|---|
| pooled r | 0.584 | 0.822 | 0.901 | 0.910 | 0.833 | 0.578 |
| failing blocks of 54 | 54 | 19 | 0 | 0 | 10 | 54 |
| verdict | FAIL | FAIL | PASS | PASS | FAIL | FAIL |

So on this data the check resolves a one-voxel surface error, but **a half-voxel error passes**.

**Practical advice:**
- Use the current release.
- If the old container image must be used, set `OMP_NUM_THREADS=1` (on our test it then passed) and check the render.

## What was already public, and what is new here
- **Already public:**
  - villa [#1588](https://github.com/ScrollPrize/villa/issues/1588): the container images (`:edge`, `:main`) ship the 2026-05-13 build; a crash was traced to them. Comments on that issue add:
    - the stale image's grower misread the volume (seed voxel 103 vs 255, areas 0.0 cm²), and "everything downstream of :edge/:main is running May code" (2026-08-25);
    - the stale image's `vc_render_tifxyz` and a current build give ink counts 5.3 % apart for `--group-idx > 0` because of a slice-step change (#1146), with the centre slice byte-identical (2026-09-27);
    - the flag `--flip-normals` is missing in the stale image (2026-09-24).
  - villa [#1982](https://github.com/ScrollPrize/villa/pull/1982): the docs point Linux users to the release AppImage and fix the tutorial's container tag.
  - villa [#1660](https://github.com/ScrollPrize/villa/issues/1660): a different silent failure, where a mesh/volume frame mismatch renders a black strip and exits 0.
- **New here:** a pixel-level comparison against the CT, on a published winner segment at `--group-idx 0` with identical flags. It shows:
  - one localised region (all 20 blocks of the left 1024 px columns) whose values do not match the CT, in every 8-thread run;
  - results that differ from run to run;
  - a pass with one thread and with the current release.

  Plus a small check that finds such regions. #1588 already documents several differences of the stale image; this adds a located, CT-referenced one.

## Why ±0.25 voxel, not offset 0
An earlier variant sampled the truth at exactly offset 0 on the mesh. On several renders it showed a dip in agreement at exactly offset 0 that vanished at ±0.25 voxel. These were false failures of the check, not renderer errors.

Averaging the two samples at ±0.25 voxel is a sub-voxel smoothing, similar in size to a renderer's own interpolation. It removes the dip and keeps the 1-voxel sensitivity shown above. We did not fully explain why the single offset-0 sample dips.

## Limitations (plainly)
- It checks **agreement of a render with the CT at the mesh** only. It says **nothing about ink**, segmentation quality or sheet switching.
- **Half-voxel errors pass.** Run `--shift-test` on your own data: smooth or flat CT lowers the sensitivity.
- **The finding covers one segment (two windows of it), one image digest and one machine.** Do not generalise beyond that.
  - The headline "exit 0, no error or warning" holds for the standard window in every captured run, but not for every window: on the second window the same image crashed with 8 threads.
  - **We have not identified the cause; the image is a 2026-05-13 build, the release a current main; we did not bisect.**
  - The container's interpolation method is not stated (the release used bicubic). That does not obviously explain it: the two agree outside the region, and the single-thread container render passes.
  - On a small number of other renders we compared (not released), the container and an older AppImage agreed bit for bit; this is not a general result.
- You need the CT (or the region of it) that the render was made from, in the same frame and resolution.
- Folds or self-intersections can give low r for legitimate reasons: mask them with `--fold`.
- The thresholds are fixed and were chosen on our renders. Treat a borderline result as a reason to look, not as proof.

## Reproduce (public inputs; about 1.5 GB download)
The commands below are the ones we ran, with the paths generalised (we used absolute paths; the flags are unchanged).
1. **Mesh:** clone `https://github.com/Nieuwlaar/pherc343-first-letters` at commit `e8370f1f88a211166056df62ea6fcfffe4a1063e` and take `outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz`.
2. **CT crop** (716 level-0 chunks, 1.50 GB; 43 s with 32 parallel requests on our connection):
   `python3 tools/fetch_ct_crop.py --mesh <…tifxyz> --out PHerc0343_L0_crop.zarr`
   This reads `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr`, with a 48-voxel margin.
3. **Render with the current release:**
   - Get the Linux AppImage of the villa GitHub release `latest`. We used `VC3D-e413746-2026-11-16-linux-x86_64.AppImage`, sha256 `8507b505ad8f18cc2224dadd54c2ff65d1538b83a21cc9773f4fbb28787efeaa`, 137.6 MB.
   - **`latest` moves:** record the commit and sha256 you get.
   - Run, e.g. via `squashfs-root/AppRun vc_render_tifxyz` after `--appimage-extract`:
   ```
   QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=8 vc_render_tifxyz --volume PHerc0343_L0_crop.zarr --group-idx 0 --scale 1 \
     --segmentation concat_w047-w048_R5B2_z9500-11000.tifxyz --num-slices 66 --slice-step 1 --cache-gb 4 \
     --crop-x 2560 --crop-y 320 --crop-width 3200 --crop-height 1280 --surface-interpolation bicubic --zarr-output render_current.zarr
   ```
   About 1-2 s with the chunks on disk; the output is 66 × 1280 × 3200 uint8.
4. **Render with the container**, twice with 8 threads and once with 1 thread (12 GB image; same flags; it has no interpolation option, so no interpolation flag is passed):
   ```
   IMG=ghcr.io/scrollprize/villa/volume-cartographer@sha256:bad516f66001abca759454cc43e4fd11e5b19aa55d36bdc2043817291c8083c4
   for T in 8 8 1; do OUT=render_docker_t${T}_$RANDOM.zarr
     docker run --rm --entrypoint "" --user $(id -u):$(id -g) -v $PWD:$PWD -w $PWD -e QT_QPA_PLATFORM=offscreen -e OMP_NUM_THREADS=$T -e HOME=/tmp $IMG \
       vc_render_tifxyz --volume PHerc0343_L0_crop.zarr --group-idx 0 --scale 1 --segmentation concat_w047-w048_R5B2_z9500-11000.tifxyz \
       --num-slices 66 --slice-step 1 --cache-gb 4 --crop-x 2560 --crop-y 320 --crop-width 3200 --crop-height 1280 --zarr-output $OUT; echo "exit $?"
   done
   ```
   About 2-3 s with 8 threads and about 10-12 s with 1 thread; each exits 0.
5. **Check each render** (a few seconds to 25 s each on CPU):
   ```
   python render_ct_check.py --mesh concat_w047-w048_R5B2_z9500-11000.tifxyz --render <render.zarr> --ct PHerc0343_L0_crop.zarr --origin 2560 320 [--shift-test] --out <name>.json
   ```
   **Expected:** the release render PASSES (0.933, 0 failing). Each 8-thread container render FAILS, with all 20 left-strip blocks failing and different blocks further right. The 1-thread container render PASSES (0.953, 0 failing).
6. **Optional:**
   - Repeat step 4 with `OMP_NUM_THREADS=2` and `4`. What we saw: non-repeatable and failing, milder with 2 threads.
   - Repeat steps 3-5 on the second window: `--crop-x 4160 --crop-y 320 --crop-width 1600 --crop-height 1280`, then check with `--origin 4160 320`. What we saw: the container with 8 threads exits 139 (crash) and leaves a partial zarr; with `OMP_NUM_THREADS=1` it exits 0 and passes; the release passes.

**Disk:** about 1.5 GB (CT crop) + about 0.3 GB per render, plus the image and the AppImage.

## Tests
`python tests/test_render_ct_check.py` takes under a minute and downloads nothing. It uses a synthetic CT and mesh:
- a correct render passes;
- a render one voxel off the surface fails;
- a render with one wrong strip fails (only that strip);
- the shift test fails at ±1 and ±2;
- a zarr render input works.

## Licence
- The code is MIT (see LICENSE).
- The CT data and the cited repository are used under their own published terms. The organisers' open-data terms apply to anything you download.
- None of that data is redistributed here.
