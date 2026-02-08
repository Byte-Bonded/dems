# Legacy Code Archive

This directory contains deprecated implementations that have been replaced with more efficient models.

## IEEE 39-Bus Triple System (Archived: February 7, 2026)

**Replaced by**: Kundur Two-Area System (`src/simulation/kundur.py`)

### Why Replaced?

The triple IEEE 39-bus system (117 buses, 30 generators) was replaced with the Kundur Two-Area System for:

1. **Simplified inter-area oscillation analysis** - Clear modal structure vs. 15-20 coupled modes
2. **PSS testing efficiency** - Single dominant 0.6 Hz inter-area mode
3. **Computational performance** - 40 states vs. 300+ states (7.5× reduction)
4. **RL training speed** - 50× faster eigenvalue analysis
5. **Analytical tractability** - Symbolic analysis possible

### Archived Files

- `ieee39bus/supergrid.py` - 117-bus tri-area supergrid implementation
- `ieee39bus/tie_lines.py` - Mesh tie-line topology between 3 areas

### Restoration

To restore the IEEE 39-bus system:
```python
from legacy.ieee39bus.supergrid import SuperGrid
grid = SuperGrid()
```

### Reference

Original implementation documented in `Grid_progress.md` (January 2026).
