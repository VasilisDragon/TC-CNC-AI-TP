# DATASET_V2 - Synthetic Label Generation

The second revision of the synthetic machining dataset introduces geometry that
better reflects real setup challenges and labels that are derived from numeric
analysis instead of hard-coded heuristics.

## Geometry generation
- Stock blanks are sampled between 60-150 mm (X), 40-120 mm (Y), and 18-70 mm (Z).
- Pockets (rectangular and circular) include random taper angles, depths, and offsets.
- Bosses, ramps, side chamfers, top shells, and vertical fillets are added with
  stochastic counts and orientations.
- When Build123d is available the generator unions or subtracts procedurally
  created wedges/bosses to introduce mixed curvature.
- STL export tolerance tightened to 0.05 mm with 0.15 rad angular tolerance.

## Feature extraction
For each mesh we sample the same scalar features used by the runtime
FeatureExtractor:
1. Axis-aligned bounding-box extents (X, Y, Z).
2. Surface area and enclosed volume.
3. Slope histogram over five bins (0-15, 15-30, 30-45, 45-60, 60-90 degrees).
4. Mean/variance of vertex-to-face normal deviation (curvature proxy).
5. Flat and steep area ratios (<15 and >=60 degrees respectively).
6. Pocket depth estimate (AABB top Z minus global min Z).

These values are stored in `meta.json` under `features_v2`, together with
supporting statistics (slope histogram, curvature moments, pocket depth,
seed). All values are rounded to six decimals for reproducibility.

## Label derivation
1. Compute area-weighted slope ratios using trimesh face normals.
2. Estimate pocket depth ratio (depth divided by part height).
3. If steep area ratio >0.35 (or >0.22 with deep pockets) -> label as
   **Waterline** with conservative stepover (18-32% of tool diameter).
4. Otherwise label as **Raster**:
   - Project face normals to the XY plane and run a weighted PCA to find the
     dominant in-plane direction.
   - Raster angle is the principal component angle (fallback to random when
     insufficient data).
   - Stepover sampled between 40-65% of tool diameter (slightly reduced for
     deep/steep mixes).
5. Confidence is reported as `1 - |0.5 - flat_ratio|` and recorded alongside the
   label for downstream filtering.
6. If OpenCAMLib bindings become available at runtime, the script marks
   `label.source = "ocl"`; in the current setup the source is `"heuristic"` but
   still driven by the geometric statistics above.

## Dataset split
After generating all samples the script shuffles the sample IDs using a seed
derived from the CLI seed, then writes:
- `train_manifest.json` - the first `train_ratio` fraction (default 80%).
- `val_manifest.json` - the remainder.

Both manifests store relative sample directory names, enabling direct
reproduction of the train/val split.

## Caveats
- PCA becomes unstable for perfectly axis-aligned or fully flat parts; the
  generator falls back to a random angle in those cases and the confidence score
  drops towards 0.5.
- Volume estimation relies on the STL being watertight. For degenerate solids we
  fall back to the convex hull volume which may slightly under/over-estimate the
  true volume.
- Build123d support is optional; when missing, the dataset will contain fewer
  mixed-curvature examples but still honours the slope-driven labelling logic.
