# Smoke Test P6

1. Launch the app and confirm the default units (mm) appear in the toolpath settings spin boxes.
2. Open View -> Units -> Inches and verify every length/feed spin box updates to "in" / "in/min" while values convert (e.g., 6 mm ~ 0.236 in).
3. Choose each tool in the Tool combo and confirm diameter/step-over/depth fields repopulate with the recommended values for that tool.
4. Close the app. Relaunch and confirm the previously selected units, tool, and parameter values (including tool overrides) are restored. The Open Model dialog should start in the last directory you used.
5. Switch back to View -> Units -> Millimeters and ensure values convert back correctly; generate a toolpath and verify the log summary uses the current unit suffix.
6. Export the toolpath twice (once per unit) and inspect the header: inch runs should start with `G20 ; units` and metric runs with `G21 ; units`, and coordinates should reflect the chosen unit system.
