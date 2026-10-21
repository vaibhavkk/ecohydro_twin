# Pune Urban Digital Twin Engine (PI-HGNN)

Physics-Informed Heterogeneous Graph Digital Twin for multi-hazard cascading stress (Hydraulics, Acoustics, Transit, Compound Shocks) along the Shivajinagar–Peth corridor.

## Repository Structure
- `app.py`: Master Streamlit Twin Dashboard with 3D GIS & Lagrangian particle engine
- `exp1_benchmark/`: Subterranean 1D Saint-Venant stormwater hydraulics
- `exp2_acoustic_benchmark/`: ISO 9613-2 street canyon acoustics & 15-year COEP validation
- `exp3_transit_civic_benchmark/`: Kasba Peth & Mandai Metro surges, Fruin LOS F bottlenecks
- `exp4_compound_coupling/`: Cross-domain cascading failure surrogate
- `data/`: Graph topologies, invert elevations, and baseline metrics

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py