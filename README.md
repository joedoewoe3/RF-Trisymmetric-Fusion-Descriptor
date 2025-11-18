# RF-Trisymmetric-Fusion-Descriptor
Combine TDOA/AOA/RSS into a one-step sensor fusion by combining in triality in symmetric geometries (e.g., 6-antenna hex arrays). Builds on full-spectrum manifolds for reference-free localization in multipath environments (900-930 MHz band focus).

## Features
- 2D/3D simulation with noise and basic multipath.
- Normalized least-squares fusion for balanced trisymmetry.
- Placeholders for SO(3) rotations (AOA symmetry) and SO(8) triality (one-step manifold unfolding via exceptional groups).

## Setup
```bash
pip install -r requirements.txt  # numpy, scipy
python src/trisym_fusion.py

