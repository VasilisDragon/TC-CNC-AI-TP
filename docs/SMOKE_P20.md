# SMOKE_P20 - Feature Vector Preview

## Preconditions
- Build with Torch/ONNX support enabled.
- Place at least one machining model in the workspace.
- (Optional) run python train/train_strategy.py --v2-features --samples 32 --epochs 2 to generate a v2-capable model card.

## Steps
1. Start the desktop app and load any STL/OBJ part.
2. Open the console pane (View -> Console) and trigger a toolpath prediction (either via the default AI or a freshly trained one).
3. Observe the info log emitted by the AI adapter (TorchAI or OnnxAI). It prints the feature vector length and the first few values, for example:
   TorchAI: feature length 17 preview [120.143, 95.882, 32.507, 84213.422, 571234.875, ...]
4. Confirm the reported length is 17 when v2 features are in use (or 6 for legacy v1 models) and the preview values are non-zero for geometry-specific slots (volume, slope bins, etc.).
5. Check that no padding/truncation warning appears for v2 models. (A warning indicates the model still expects the legacy 6-D input.)

## Expected
- Console shows a single feature-preview log per AI session with the correct dimensionality.
- No warnings are printed when the model schema matches the v2 feature count.
- Toolpath generation completes successfully after logging the preview.
