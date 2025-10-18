# Smoke Test P8

1. Launch the app and load a sizeable mesh via File -> Open Model.... A progress dialog should appear; move other windows or orbit the viewer to confirm the UI stays responsive.
2. While the import dialog counts up, click Cancel. Verify the dialog closes, the Console logs "Import cancelled.", and the viewer remains responsive.
3. Re-import the same mesh and allow it to finish. Confirm the Console logs the import duration and the model appears.
4. Click "Generate Toolpath". A generation progress dialog should appear; orbit/pan the model to ensure rendering stays at 60 FPS.
5. Cancel generation mid-way and verify the dialog closes, the Console shows "Toolpath generation cancelled.", and no new path is applied.
6. Run "Generate Toolpath" again and let it finish. Confirm the Console logs the duration and AI decision, and the toolpath overlay updates.
