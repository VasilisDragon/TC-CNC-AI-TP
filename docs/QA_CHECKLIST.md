# QA Checklist

1. Install the latest NSIS build artifact and launch AIToolpathGenerator; confirm the first-run welcome dialog appears and the sample STL loads in the viewport.
2. Close and relaunch the app; ensure the welcome prompt does not recur and the most recently opened model is restored when available.
3. Observe the status bar to verify GPU name, AI backend summary, and FPS counter update within five seconds of startup.
4. Toggle each camera preset (Top, Front, Right, Iso) and confirm the viewport reorients correctly without throwing errors.
5. Use File -> Open to load `samples/sample_part.stl`; confirm the model browser shows file name, triangle count, and bounds.
6. Import a large third-party STL or OBJ and ensure the progress dialog can be canceled mid-import.
7. Switch to an invalid or empty STL file and verify the user receives an "Import Failed" warning dialog.
8. Generate a toolpath with default parameters; confirm the console log, segment count, and simulation tool follow the path.
9. Adjust step-over and feed values in the Toolpath Settings panel and regenerate; verify settings persist on subsequent runs.
10. Export the generated toolpath to G-code and re-open the preview tab to ensure the preview displays the first 100 lines.
11. Attempt to export without a generated toolpath; the app should show a friendly informational dialog.
12. Start a toolpath generation and cancel via the progress dialog; confirm the UI returns to idle without a crash.
13. Switch between Torch and ONNX models (if available) in the AI combo box and confirm device labels, status bar text, and logging all update.
14. Enable the "Force CPU" option in AI Preferences and ensure the device label reflects the change immediately.
15. Verify that switching units between millimeters and inches updates the toolpath settings UI and the log.
16. Reset the camera via the View menu and ensure the viewport re-centers on the active model.
17. Start, pause, and stop the simulation toolbar controls to validate feed progress and slider scrubbing.
18. Drag the simulation progress slider during playback to scrub to different points without freezing the UI.
19. Run the bundled sample tour again by deleting the `ui/firstRunCompleted` flag from settings and relaunching; confirm the tour runs only once.
20. Open multiple models sequentially and verify the console history and status bar maintain accurate information for the active file.
21. Check that closing the app while a generation is running shuts down gracefully and no background threads remain.
22. Inspect `models/` and `samples/` directories in the installed location to ensure required assets are present.
23. Trigger an AI backend failure (e.g., remove model file) and confirm fallback warnings appear without crashing.
24. Open the About dialog and validate the version, commit hash, build type, backend list, and GPU summary string.
25. Uninstall the application via Programs and Features and confirm install directory and start menu shortcuts are removed.