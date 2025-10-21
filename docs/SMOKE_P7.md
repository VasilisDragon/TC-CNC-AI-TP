# Smoke Test P7

1. Start the application and confirm the AI combo in the Toolpath dock shows "Default (built-in)".
2. Add a dummy model file (e.g., copy `resources/tools.json` to `models/foo.pt`) or use one already listed; open AI > Select Model... and choose it. The dock combo should update to the selected file name.
3. Generate a toolpath. In the Console, verify a log entry like `AI: Using foo.pt, steps: [1] Raster rough step-over=3.000 mm step-down=1.000 mm angle=45.0 deg; [2] Raster finish step-over=1.500 mm step-down=0.500 mm angle=45.0 deg` appears.
4. Switch back to the default model using the dock combo and generate again; the log should reflect `Default` with the same step sequence.
5. Restart the app to confirm the previously selected model is restored and the AI combo reflects the saved choice.
