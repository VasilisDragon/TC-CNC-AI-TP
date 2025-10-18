# SMOKE_P16 - Stock & Machine

1. Launch the app and load any sample model (e.g. samples/sample_part.stl).
2. In **Stock & Machine**:
   - Choose **Origin Mode: Model Minimum**.
   - Set **Margin** to 2.0 mm (width/length/height update automatically from the model AABB).
3. Open **Machine -> Presets -> GRBL Router** to load GRBL machine limits (rapid/safe/clearance).
4. Generate a toolpath.
5. Observe in the viewport:
   - The first motion lifts to the configured safe Z before moving.
   - Horizontal travel runs at clearance Z and is drawn with the lighter rapid style, while vertical rapids use the bright rapid style.
6. Export G-code (optional) and confirm header includes the machine info with GRBL feeds.
