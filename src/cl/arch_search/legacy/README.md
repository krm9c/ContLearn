# Legacy Architecture Search Functions

This directory contains deprecated architecture search functions that are no longer used in the codebase.

## Why these functions were deprecated

The functions in this directory (`arch_search_CNN` and `arch_search_CNN3D`) used **hardcoded constants** from `constants.py` instead of reading configuration parameters from config files.

For example, they used:
- `og_epochs = DEFAULT_CNN_ARCH_SEARCH_EPOCHS` (hardcoded)
- `optim = optax.adam(DEFAULT_ARCH_SEARCH_LR)` (hardcoded)
- `threshold = DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD` (hardcoded)

This meant that users could not control architecture search behavior through config files.

## Current implementation

The current codebase uses **`arch_search_CNN_fresh`** (in `../cnn_search.py`) which:
- ✅ Reads all parameters from config files with fallbacks to constants
- ✅ Gives users full control over architecture search hyperparameters
- ✅ Works for both CNN and CNN3D architectures

## Active functions that properly respect config

| Function | File | Config Parameters Used |
|----------|------|------------------------|
| `arch_search_CNN_fresh` | `cnn_search.py` | `arch_search_epochs`, `arch_search_lr`, `arch_search_batch_size`, `arch_search_exp_replay`, `arch_search_loss_threshold`, etc. |
| `arch_search_MLP` | `mlp_search.py` | Same as above |
| `arch_search_GCN` | `gcn_search.py` | Same as above |

## Should I use these legacy functions?

**NO.** These functions are kept only for historical reference. They are not imported or used anywhere in the codebase.

If you need CNN architecture search, use `arch_search_CNN_fresh` from `../cnn_search.py`.

---

*Moved to legacy on 2025-12-14 to clean up codebase and eliminate confusion about which functions respect config file parameters.*
