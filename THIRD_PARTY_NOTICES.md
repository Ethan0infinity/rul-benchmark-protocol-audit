# Third-Party Notices

## Dual-Mixer architecture adapter

`src/rul/models.py` contains an adapter of the Dual-Mixer layer equations from:

- En Fu, Yanyan Hu, Kaixiang Peng, and Yuxin Chu, "Supervised contrastive learning based dual-mixer model for Remaining Useful Life prediction," *Reliability Engineering & System Safety*, 251, 110398 (2024), DOI: 10.1016/j.ress.2024.110398.
- Official repository: `https://github.com/fuen1590/PhmDeepLearningProjects`
- Audited commit: `727c020cabba2c6ae96e8f0e28f7f0121b292e81`
- Upstream license: MIT, Copyright (c) 2024 fuen1590.

The local adapter preserves the published repository's Dual-Mixer structure and default non-FSGRI setting, but uses this project's locked engine split and result interface. It is therefore described as an **official-code-derived architecture adapter**, not a bitwise native reproduction. The full upstream MIT license is available in the repository above and must accompany any redistributed substantial copy.
