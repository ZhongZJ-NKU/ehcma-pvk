# MSEC Analysis Report

## 1. Descriptor-Level Mechanisms
| Feature | Evidence | Direction | Optimal Range |
|---------|----------|-----------|---------------|
| Bandgap | abs_mean=0.875, mean=+0.057, std=1.078, pos_ratio=0.25; dependence monotonic decreasing, positive near 1.67-1.69, negative above 1.71 | narrow low bandgap favorable; higher values decrease predicted PCE | around 1.67-1.69 |
| HBA_total | abs_mean=0.108, mean=+0.003, std=0.126, pos_ratio=0.72; dependence threshold positive above 8 | higher HBA_total above threshold increases predicted PCE | >8 |
| HOMO_LUMO_gap | abs_mean=0.100, mean=-0.007, std=0.120, pos_ratio=0.62; dependence positive below 4.2 and negative above 4.5 | smaller HOMO-LUMO gap favorable; large gap reduces predicted PCE | <4.2 |
| HOMO_avg | abs_mean=0.078, mean=-0.003, std=0.103, pos_ratio=0.68; dependence positive above -5.5 and negative below -5.5 | shallower HOMO increases predicted PCE | >-5.5 |
| HeteroSym/Mixture/MACl_additive | abs_mean=0.090/0.050/0.131, mean=+0.002/-0.001/-0.011, std=0.097/0.061/0.182, pos_ratio=0.44/0.77/0.71; dependence positive at HeteroSym=1, Mixture=1, MACl_additive=1 | heterogeneous symmetry, mixing, and additive assembly favorable | HeteroSym=1, Mixture=1, MACl_additive=1 |

## 2. Substructure-Level Mechanisms

| Fragment Group | KRFP Indices | ACIN Weight Range | Contribution | Role Hypothesis |
|----------------|--------------|-------------------|--------------|-----------------|
| Nitrogen/sulfur-containing aromatics and polar anchors | krfp_1870, krfp_3999, krfp_1153, krfp_2511, krfp_4256, krfp_4852, krfp_2854, krfp_3127 | 0.474-0.718 | high_contrib | Likely dominant buried-interface motifs; may support energy alignment, polarity, and co-assembly |
| Supportive fused aromatics and polar spacers | krfp_2641, krfp_1810, krfp_3881, krfp_1249, krfp_3601, krfp_2070, krfp_3891, krfp_4837, krfp_1641 | 0.285-0.463 | high_contrib | Secondary supportive motifs; may contribute to HBA/dipole passivation and packing |
| Context-dependent flexible linkers and polar groups | krfp_3223, krfp_2778, krfp_1304, krfp_1949, krfp_4100, krfp_2643, krfp_1714, krfp_1200, krfp_3382, krfp_1815, krfp_50, krfp_4267, krfp_2638, krfp_2017, krfp_1747, krfp_1771 | 0.157-0.265 | high_contrib | Weak or context-dependent motifs; may tune morphology but insufficient alone |
| Low-contrib weak or antagonistic motifs | krfp_73, krfp_3905, krfp_4670, krfp_3692, krfp_101, krfp_2685, krfp_1251, krfp_4607, krfp_3288 | 0.112-0.141 | low_contrib | Weak or antagonistic motifs; avoid as primary design handles |

## 3. Synergistic Mechanisms
Do not use Bandgap as a descriptor in any mechanism below.

1. **Mechanism**: Energy-level alignment at the buried interface
   - Descriptor evidence: HOMO_avg threshold positive above -5.5, abs_mean=0.078, pos_ratio=0.68; HOMO_LUMO_gap threshold positive below 4.2, abs_mean=0.100, pos_ratio=0.62. Evidence suggests shallower HOMO and smaller gap may improve hole extraction.
   - Substructure evidence: top-tier high-ACIN cluster, especially krfp_1870, krfp_1153, krfp_4852, and krfp_2854, may encode dominant electronic motifs.
   - Confidence: medium

2. **Mechanism**: Polarity and HBA-mediated passivation
   - Descriptor evidence: HBA_total positive above 8, abs_mean=0.108, pos_ratio=0.72; Dipole_avg shows peaks at 1.8, 3.5, and 5.2; TPSA_avg performs best near 64 within a 60-68 window; Rho_avg positive above 1.38. Evidence suggests polar or HBA-rich motifs may passivate traps and tune interfacial dipoles.
   - Substructure evidence: top-tier and mid-tier high-ACIN clusters, including krfp_4852, krfp_2854, krfp_3881, krfp_1810, and krfp_3891, may supply supportive polar motifs.
   - Confidence: medium

3. **Mechanism**: Heterogeneous symmetry-driven mixed co-assembly
   - Descriptor evidence: HeteroSym positive at 1, abs_mean=0.090; Mixture positive at 1, abs_mean=0.050, pos_ratio=0.77; MACl_additive positive at 1, abs_mean=0.131, pos_ratio=0.71. Evidence suggests heterogeneous symmetry matching in mixed assembly may improve coverage and packing.
   - Substructure evidence: top-tier high-ACIN motifs such as krfp_1870, krfp_2511, and krfp_3127 may support co-assembly, while mid-tier motifs may assist interface organization.
   - Confidence: medium

4. **Mechanism**: Balanced volume and conjugated packing
   - Descriptor evidence: Vol_ratio positive above 1.2 with peak around 1.5, abs_mean=0.068, pos_ratio=0.26; sp2_total positive above 45, abs_mean=0.030, pos_ratio=0.63; Rho_avg positive above 1.38. Evidence suggests balanced molecular volume and conjugated rigidity may improve morphology.
   - Substructure evidence: mid-tier and lower high-ACIN clusters, including krfp_1641, krfp_3223, and krfp_2778, may provide packing or tuning motifs, but lower weights suggest supportive roles only.
   - Confidence: low

## 4. Design Recommendations
### Recommended Substructures (Evidence-Based Top 10)
| Substructure | ACIN Weight | Reason |
|--------------|-------------|--------|
| krfp_1870 | 0.71835905 | Highest ACIN weight; likely dominant motif for energy-level alignment and co-assembly packing |
| krfp_1153 | 0.6354857 | Second-highest ACIN weight; contributes to polarity and energy-level tuning |
| krfp_2511 | 0.58772534 | Strong high-ACIN motif; aids heterogeneous symmetry matching and interface packing |
| krfp_4852 | 0.491145 | Strong high-ACIN motif; drives energy alignment and polarity synergy |
| krfp_2854 | 0.48723269 | Supports HBA_total threshold and interface dipole |
| krfp_3881 | 0.42510873 | Supports frontier-orbital alignment and structural heterogeneity |
| krfp_3891 | 0.28930128 | Assists molecular packing and polarity passivation |
| krfp_1641 | 0.28592458 | Promotes mixed co-assembly and surface coverage |
| krfp_3223 | 0.26414782 | Provides spatial complementarity and coverage |
| krfp_2778 | 0.19918147 | Supports anchoring interactions and volume balance |

### Favorable Combinations
- **Combination**: Energy-alignment core
  - KRFP indices: krfp_1870, krfp_1153, krfp_4852
  - Reason: high-ACIN motifs likely support favorable HOMO_avg and HOMO_LUMO_gap trends
  - Supporting evidence: HOMO_avg positive above -5.5 and HOMO_LUMO_gap positive below 4.2
- **Combination**: Polarity/passivation group
  - KRFP indices: krfp_2511, krfp_2854, krfp_3881, krfp_3891
  - Reason: high-ACIN motifs may provide HBA-rich or polar interface passivation and packing
  - Supporting evidence: HBA_total positive above 8, Dipole_avg peaks at 1.8, 3.5, 5.2, TPSA_avg best near 64 (60-68), Rho_avg positive above 1.38
- **Combination**: Co-assembly/packing group
  - KRFP indices: krfp_1641, krfp_3223, krfp_2778
  - Reason: high-ACIN motifs may support heterogeneous symmetry matching and balanced packing
  - Supporting evidence: HeteroSym=1, Mixture=1, MACl_additive=1, and Vol_ratio peak around 1.5

### Avoided Combinations
- **Combination**: Low-weight dilution cluster (krfp_73, krfp_3905, krfp_4670)
  - Reason: weak low-contrib motifs may dilute top high-ACIN interface signals
  - Supporting evidence: ACIN weights 0.141, 0.141, and 0.141 with low_contrib labels
- **Combination**: Low-weight polarity mismatch cluster (krfp_101, krfp_2685, krfp_1251, krfp_4607, krfp_3288)
  - Reason: low weights and low_contrib suggest insufficient support for favorable polarity or passivation
  - Supporting evidence: ACIN weights 0.123, 0.118, 0.113, 0.112, and 0.112 with low_contrib labels