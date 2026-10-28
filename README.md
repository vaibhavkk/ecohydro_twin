# Physics-Informed Heterogeneous Graph Digital Twin for Compound Urban Hazards (`ecohydro_twin`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric: 2.4](https://img.shields.io/badge/PyG-2.4-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![Engine: EPA--SWMM 5.2](https://img.shields.io/badge/SWMM-5.2-brightgreen.svg)](https://www.epa.gov/water-research/storm-water-management-model-swmm)

This repository hosts the official open-source numerical implementation, dataset configurations, and evaluation benchmarks for the *Building and Environment* Short Communication:

> **"Compound Hydro-Acoustic and Egress Hazards in Historical Street Canyons: A Physics-Informed Relational Network Formulation"**  
> *Vaibhav K. Khatavkar, Mahesh Shindikar, and Parag Sadgir*  
> COEP Technological University, Pune 411005, Maharashtra, India.

---

## 📌 Scientific Background & Core Findings

Traditional municipal disaster mitigation workflows assess open-channel stormwater drainage (e.g., US EPA SWMM), street canyon acoustics (ISO 9613-2), and pedestrian crowd kinematics (LWR/Cellular Automata) as independent systems. 

In high-density vernacular commercial corridors—such as the 1.8-km historical spine of Pune, India ($H/W \in [0.85, 3.25]$, rights-of-way $3.2\,\mathrm{m}\dots 12.0\,\mathrm{m}$)—intense convective cloudbursts coinciding with major street assemblies create severe cross-domain feedback loops:
1. **Stormwater Surcharge & Inundation:** Subterranean conduits surcharge through catch basins, ponding up to $0.38\,\mathrm{m}$ of surface runoff.
2. **Hydrodynamic Wading Retardation:** Walking velocity decays exponentially with standing water depth ($v_{\mathrm{eff}} = v(k) \cdot e^{-2.5 h_{\mathrm{pond}}}$), causing crowd densities to reach jam capacity ($4.92\,\mathrm{pax/m^2}$).
3. **Debris Inlet Choking:** Moving pedestrian crowds wash litter over catch basins, causing up to $70\%$ grate blinding.
4. **Canyon Acoustic Masking:** Deep masonry facades trap processional sound energy ($>108\,\mathrm{dBA}$), drowning out municipal emergency broadcast sirens ($\Delta L \le -6\,\mathrm{dB}$).

**Core Finding:** Uncoupled simulation frameworks underestimate corridor evacuation clearance times by **$42.6\%$** ($14.2\,\mathrm{min}$ predicted vs. $24.8\,\mathrm{min}$ actual). This Physics-Informed Heterogeneous Graph Neural Network (PI-HGNN) resolves these interfacial feedbacks in **$8.85\,\mathrm{ms}$** per inference step on standard academic workstations while restricting boundary mass continuity discrepancy to **$1.18\%$**.

---

## 🗂️ Repository Structure

```text
ecohydro_twin/
│
├── data/                                 # Calibrated network topologies and forcing data
│   ├── swmm_corridor.inp                 # US EPA SWMM 5.2 corridor dynamic routing model
│   ├── imd_idf_hyetographs.csv           # Pune IMD convective cloudburst profiles (40-110 mm/h)
│   ├── corridor_geometry_nodes.csv       # Node coordinates, widths, and facade aspect ratios
│   └── acoustic_emission_sources.csv     # Line-source sound power levels & spectra
│
├── notebooks/                            # Interactive execution & figure reproductions
│   ├── 01_hydro_acoustic_coupling.ipynb  # Reproduces coupled timeline & Figure 2
│   └── 02_pi_hgnn_evaluation.ipynb       # Mass continuity ablation & performance metrics
│
├── exp1_benchmark/                       # EPA-SWMM dynamic wave routing runs & verification
├── exp2_acoustic_benchmark/              # ISO 9613-2 / Kang (2007) canyon radiosity scripts
├── exp3_transit_civic_benchmark/         # Lax-Friedrichs finite-volume crowd kinematics
├── exp4_compound_coupling/               # End-to-end PI-HGNN model checkpoints & training loops
│
├── app.py                                # Streamlit demonstration dashboard
├── app-hydro.py                          # Dedicated hydrodynamic surrogate dashboard
├── app-sound1.py                         # Canyon acoustic radiosity visualization
│
├── requirements.txt                      # Complete Python environment dependencies
└── README.md                             # Project overview and reproduction guidelines