# SMOKE_P19 – Multi‑pass Scheduler

## Setup
- Launch the CNCTC desktop app with any valid 3D model available (the bundled sample is fine).
- Reset the application settings if previous runs modified toolpath options.

## Steps
1. Open a small STL/OBJ model and wait for the geometry to load.
2. In the *Toolpath Settings* dock, ensure the new **Pass Planning** section is visible.
3. Leave both *Enable roughing pass* and *Enable finishing pass* checked, set *Stock allowance* to `0.4` mm (or `0.016` in), and set *Ramp angle* to `5` deg.
4. Choose any flat endmill tool and click **Generate Toolpath**.
5. After generation completes, inspect the passes in the viewer and the console log.

## Expected
- The console banner references two passes and lists the roughing allowance that was preserved before finishing.
- The simulated toolpath shows initial segments that stop short of the final surface (rough) followed by a finishing pass that removes the allowance.
- Entry and exit motions are diagonal ramps rather than straight plunges, and all rapid links retract to clearance before traversing.
- The *Toolpath Settings* controls remember the allowance, ramp angle, and pass toggles across unit changes and application restarts.
