# Smoke Test P13

1. Load a model and generate a toolpath (with or without OCL). Wait until the console reports the strategy and the toolpath appears in the viewer.
2. In the **Simulation** toolbar at the bottom, click **Play**. A small green sphere should travel along the toolpath, following cut moves at the configured feed rate. Rapid moves display in yellow.
3. Use the progress slider to scrub forward and backward; the tool should jump to the corresponding position without stutter. Verify the console log continues to update smoothly.
4. Adjust the speed slider between 0.25× and 4×. Confirm playback accelerates and decelerates while remaining smooth.
5. Press **Pause** to freeze the simulation, then **Play** to resume from the same point. Press **Stop** and ensure the tool returns to the program start and the slider resets.
6. Regenerate the toolpath and confirm the simulation resets automatically. Play again to ensure the glyph follows the updated path.
