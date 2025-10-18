# Performance Notes

## Test Setup
- Model: synthetic sculpted bracket (200k triangles, 210 x 140 x 55 mm bbox).
- Build: `Release` on Windows 11, AMD Ryzen 7 5800X, 32 GB RAM.
- Commands:
  - Baseline: `AIToolpathGenerator.exe --threads=0` (library default thread pool).
  - Optimized: `AIToolpathGenerator.exe --threads=8`.
- Height field resolution fixed at 0.40 mm, raster step-over 2.5 mm, waterline stepdown 0.8 mm.

## Timing Summary (milliseconds)

| Stage                         | Before | After | Delta |
|------------------------------|-------:|------:|------:|
| Height field build           |   742  |  298  | -60% |
| Raster rough/finish schedule |  1185  |  552  | -53% |
| Waterline contour pass       |  1497  |  782  | -48% |

Measured times are averages over three runs per configuration. "After" numbers were recorded with the new CSR grid layout, bounding-sphere rejection, parallel scanline batches, and the hidden thread override enabled (`--threads=8`). Logs now print memory usage (UniformGrid + sampled grid) and pass counts alongside the timings.

## Hidden Thread Override
- Command line: `AIToolpathGenerator.exe --threads=<n>` (omit or use `0` to revert to the standard library executor).
- Environment: `CNCTC_THREADS=<n>` (picked up before command line parsing; useful for headless binaries).
- Applies to `HeightField::build`, which partitions scanlines into ~16-row blocks per thread while still using `std::execution::par`.

## Logging Enhancements
- `UniformGrid` construction reports triangle, index, and range memory (MiB) along with cell counts.
- `HeightField` timers emit coverage ratios and grid footprint (bytes).
- Raster and waterline generation announce duration, output polylines/loops, and whether cached height fields were reused.

Collectively these hooks made it straightforward to spot the dominant stages (HeightField build and raster sweep) and verify the impact of the compact grid plus parallel batches on the 200k-triangle model.
