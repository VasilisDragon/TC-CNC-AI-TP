# Toolpath Safety Notes

- Waterline passes invoke `tp::GougeChecker` to ensure each segment respects the configured leave-stock. The checker samples tool centerlines against the mesh with a lightweight XY grid and raises passes (clamped by machine `safeZ`) whenever the requested clearance is not met.
- Loops that cannot satisfy clearance without breaching `safeZ` are skipped and reported in the generation log. Review `docs/map.md` for the module layout and `tp/GougeChecker.*` for implementation details.
- The Toolpath Settings panel exposes a *Leave Stock* control that maps to `tp::UserParams::leaveStock_mm`; roughing allowances continue to use `stockAllowance_mm`, which is kept in sync with the UI value for compatibility.
