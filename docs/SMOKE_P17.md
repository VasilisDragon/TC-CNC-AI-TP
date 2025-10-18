# SMOKE_P17 - HeightField Raster

1. Launch the app and open a model with visible relief (e.g. samples/sample_part.stl).
2. In **Toolpath Settings**:
   - Ensure Use HeightField (software raster) is checked.
   - Set Raster Angle to 30 deg and Max Depth/Pass to 0.5 mm to exercise multi-pass slicing.
3. Generate the toolpath.
   - The console should log **Height field built (...)** with sample counts and build time.
   - Inspect the preview; the first pass follows the surface rather than a flat plane.
4. Generate again without changing parameters.
   - The console should now report **Height field cache hit (...)** instead of rebuilding.
5. Start the simulation and verify:
   - Rapids lift to safe Z, link moves traverse at clearance Z.
   - Multiple depth passes are visible along the raster because of the 0.5 mm limit.
