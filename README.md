# H3-Oscillator

Internal research: a gauge-equivariant convolutional network with continuous-time bounded oscillator dynamics, operating natively on H3 hexagonal cells.

**Current status:** Phase 0 complete (Gray-Scott data generator + hand-crafted baseline B1).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.phase0_sanity_check
python -m scripts.phase0_b1_verification
```

## Project structure

| Path | Purpose |
|---|---|
| `src/` | Core modules (H3 region, Gray-Scott dynamics, visualization) |
| `scripts/` | Phase-specific runner scripts |
| `figures/` | Generated visualizations |
| `data/` | Generated datasets (gitignored) |
| `PHASE_0_STATUS.md` | Current implementation status |

## Foundational papers

- Cohen et al. 2019, *Gauge Equivariant Convolutional Networks and the Icosahedral CNN*
- Hasani et al. 2021, *Liquid Time-constant Networks*
- Hoogeboom et al. 2018, *HexaConv*

## License

Internal research — not yet for public distribution.
