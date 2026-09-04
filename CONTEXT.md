# Beach Tracker Context

## Core Terms

- **Support point**: A labeled image point that lies on a world line. It is not a corner by default.
- **Intersection**: A labeled point shared by more than one world line. Some intersections are true court corners, but not all.
- **Court support polyline**: The ordered set of support points the user clicks on visible court lines in the corrected view.
- **Corrected view**: The image after the initial horizontal-line warp correction and rotation. This is the working view used for court, net, and ball labeling.
- **Initial pinhole-like**: A replay of the corrected view used during labeling. It is a bootstrap view, not the final calibrated camera.
- **True pinhole**: The later camera-model result after scene-geometry refinement from court, net, and ball observations.
- **Camera model**: Intrinsics, pose, and correction terms that map between image pixels and world geometry.
- **Solver path**: The ordered sequence of improving states produced by a deterministic refinement solve.

## Geometry Interpretation

- Court clicks are interpreted as support points on court lines unless the same point is explicitly shared by multiple lines.
- Net observations are sparse indicators that should be refined into antenna lines and a net-top curve by deterministic geometry.
- Ball labels are coarse hints that should be refined into center and radius by deterministic image analysis.
