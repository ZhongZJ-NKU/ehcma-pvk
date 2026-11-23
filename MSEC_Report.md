# MSEC Analysis Report

## 1. Descriptor-Level Mechanisms

Bandgap is treated only as a device screening condition (1.67 to 1.70, positive below about 1.71) and is not used as a molecular mechanism.

| Feature | Evidence | Direction | Optimal Range |
|---------|----------|-----------|---------------|
| HeteroSym / Mixture / MACl_additive | abs_mean=0.090/0.050/0.131, mean=+0.002/-0.001/-0.011, std=0.097/0.061/0.182, pos_ratio=0.44/0.77/0.71; dependence positive at HeteroSym=1, Mixture=1, MACl_additive=1 | heterogeneous symmetry, mixing and additive assembly favorable | HeteroSym=1, Mixture=1, MACl_additive=1 |
| HBA_total | abs_mean=0.108, mean=+0.003, std=0.126, pos_ratio=0.72; threshold dependence. Evidence suggests a moderate acceptor count supports passivation. | positive above about 8 | >8 |
| HOMO_avg / HOMO_LUMO_gap | HOMO_avg abs_mean=0.078, mean=-0.003, std=0.103, pos_ratio=0.68; HOMO_LUMO_gap abs_mean=0.100, mean=-0.007, std=0.120, pos_ratio=0.62. Evidence suggests a shallower HOMO and a smaller gap may improve hole extraction. | HOMO_avg positive above about -5.5; HOMO_LUMO_gap positive below about 4.2 and negative above about 4.5 | HOMO_avg >-5.5; HOMO_LUMO_gap <4.2 |
| Vol_ratio / Rho_avg | Vol_ratio abs_mean=0.068, pos_ratio=0.26; Rho_avg abs_mean=0.045, pos_ratio=0.61. Evidence suggests balanced molecular volume with a higher density may improve packing. | Vol_ratio positive above about 1.2 with a peak near 1.5; Rho_avg positive above about 1.38 | Vol_ratio near 1.5 (>1.2); Rho_avg >1.38 |
| Dipole_avg / TPSA_avg | Dipole_avg abs_mean=0.036, pos_ratio=0.50; shows peaks at 1.8, 3.5 and 5.2. TPSA_avg performs best near 64 within a 60-68 window, indicating a multi-peak polarity preference rather than a monotonic trend. | multi-peak for Dipole_avg; window for TPSA_avg | Dipole_avg 1.8 / 3.5 / 5.2; TPSA_avg 60-68 (best 64) |
| sp2_total | abs_mean=0.030, mean=+0.003, pos_ratio=0.63; threshold dependence. Evidence suggests sufficient sp2 content supports pi stacking and rigid packing. | positive above about 45 | >45 |

## 2. Substructure-Level Mechanisms

| Fragment Group | KRFP Indices | ACIN Weight Range | Contribution | Role Hypothesis |
|----------------|--------------|-------------------|--------------|-----------------|
| fused_n_heteroarene, N-substituted tricyclic | krfp_1870 | 0.718 | positive | Rigid fused N-heteroaromatic surface; may tune energy levels (HOMO_avg above -5.5) and support pi packing without adding polarity. |
| alkoxyarene, terminal methoxy | krfp_3999 | 0.641 | positive | Aryl ether oxygen acts as a hydrogen-bond acceptor and mild electron donor, consistent with HBA_total above 8. |
| quaternary_carbon, tetra-substituted | krfp_1153 | 0.635 | positive | Tetra-substituted carbon providing a steric node for volume matching (Vol_ratio near 1.5); a spacer rather than an electronic handle. |
| fused_n_heteroarene, N-linked tricyclic | krfp_2511 | 0.588 | positive | Electron-rich N-linked tricyclic; supports heterogeneous symmetry (HeteroSym = 1) and interfacial packing. |
| carboxylic_acid, aniline | krfp_4256 | 0.583 | positive | Acid/aniline donor-acceptor pair; strong hydrogen bonding for passivation, consistent with HBA_total above 8. |
| thioether, aryl | krfp_4852 | 0.491 | positive | Polarizable aryl sulfur linking the aromatic core to the interface; consistent with the polarity windows (TPSA_avg 60-68, Dipole_avg peaks). |
| thioether, dialkyl | krfp_2854 | 0.487 | positive | Aliphatic sulfur acceptor; contributes interfacial dipole but also sp3 flexibility, so it is best used as a minor component. |
| fused_n_heteroarene, N-H tricyclic | krfp_3127 | 0.474 | positive | N-H carbazole; adds hydrogen-bond donor character while retaining rigid pi character. |
| amide/alcohol with bromothiophene | krfp_2641 | 0.463 | positive | Mixed acceptor (amide, alcohol, bromothiophene); may support local polarity but is structurally heterogeneous. |
| alkoxyarene, ring methoxy | krfp_1810 | 0.457 | positive | Ring-attached methoxyarene; ether acceptor plus aryl pi character, secondary to krfp_3999. |
| amine, alkyl C-N | krfp_3881 | 0.425 | positive | Aliphatic amine: its nitrogen lone pair raises local polarity as a hydrogen-bond acceptor, with no pi contribution; a supporting polar handle rather than an electronic one. |
| aniline, methyl-substituted | krfp_3601 | 0.406 | positive | Aniline nitrogen acting as acceptor/donor at the interface; supports polarity without added rigidity. |
| fused_n_heteroarene, N-substituted | krfp_2070 | 0.383 | positive | Substituted carbazole-type system; contributes aromatic pi character and additional grafting points. |
| sulfonic_acid, imine/hydrazine | krfp_4837 | 0.351 | positive | Strongly acidic function; high polarity that may over-polarize the interface if used repeatedly. |
| aniline, N,N-dimethyl | krfp_3891 | 0.289 | positive | Tertiary aniline; polar but a weak hydrogen-bond donor, so a supporting rather than a leading role. |
| arene, disubstituted benzene | krfp_1641 | 0.286 | positive | Rigid aromatic spacer; one of the fragments that makes an asymmetric/symmetric pair complementary rather than merely additive. |
| alkane_chain, aliphatic C-C | krfp_3223 | 0.264 | positive | Flexible C-C linker; provides spatial complementarity at the cost of sp3 flexibility (sp2_total above 45 favours rigidity). |
| imine/amine, fused ring | krfp_4100 | 0.211 | positive | Cyclic imine/amine; may add polarity and moderate rigidity to the interface layer. |
| amide/alcohol, cyclic | krfp_2643 | 0.202 | positive | Cyclic amide/alcohol; acceptor-rich and moderately rigid, supporting local polarity. |
| phosphonate_ester | krfp_2778 | 0.199 | positive | Phosphonate ester; P=O chemistry may echo the anchoring group, but its low weight limits it to a supportive role. |

## 3. Synergistic Mechanisms

1. **Mechanism**: Energy-level alignment at the buried interface.
   - Descriptor evidence: HOMO_avg positive above -5.5 (abs_mean=0.078, mean=-0.003, std=0.103, pos_ratio=0.68); HOMO_LUMO_gap positive below 4.2 (abs_mean=0.100, mean=-0.007, std=0.120, pos_ratio=0.62). Evidence suggests a shallower HOMO and a smaller gap may improve hole extraction.
   - Substructure evidence: the top-tier positive-weight cluster, especially krfp_1870, krfp_4852 and krfp_2854, may encode electronic motifs, while krfp_1153 contributes steric volume rather than electronic structure.
   - Confidence: medium

2. **Mechanism**: Polarity and HBA-mediated passivation.
   - Descriptor evidence: HBA_total positive above 8 (abs_mean=0.108, mean=+0.003, std=0.126, pos_ratio=0.72); Dipole_avg peaks at 1.8, 3.5 and 5.2; TPSA_avg performs best near 64 within a 60-68 window; Rho_avg positive above 1.38. Evidence suggests polar or HBA-rich motifs may passivate traps and tune interfacial dipoles.
   - Substructure evidence: top-tier and mid-tier positive-weight clusters, including krfp_4852, krfp_2854, krfp_3881, krfp_1810 and krfp_3891, may supply supportive polar motifs.
   - Confidence: medium

3. **Mechanism**: Heterogeneous symmetry-driven mixed co-assembly.
   - Descriptor evidence: HeteroSym positive at 1 (abs_mean=0.090, std=0.097); Mixture positive at 1 (abs_mean=0.050, mean=-0.001, std=0.061, pos_ratio=0.77); MACl_additive positive at 1 (abs_mean=0.131, mean=-0.011, std=0.182, pos_ratio=0.71). Evidence suggests heterogeneous symmetry matching in mixed assembly may improve coverage and packing.
   - Substructure evidence: top-tier motifs such as krfp_1870, krfp_2511 and krfp_3127 may support co-assembly, while mid-tier motifs may assist interface organization.
   - Confidence: medium

4. **Mechanism**: Balanced volume and conjugated packing.
   - Descriptor evidence: Vol_ratio positive above 1.2 with a peak near 1.5 (abs_mean=0.068, pos_ratio=0.26); sp2_total positive above 45 (abs_mean=0.030, mean=+0.003, pos_ratio=0.63); Rho_avg positive above 1.38. Evidence suggests balanced molecular volume and conjugated rigidity may improve morphology.
   - Substructure evidence: mid-tier and lower positive-weight clusters, including krfp_1641, krfp_3223 and krfp_2778, may provide packing or tuning motifs, but their lower weights suggest supportive roles only.
   - Confidence: low

## 4. Design Recommendations

### Recommended Substructures (Evidence-Based Selection)

| Substructure | ACIN Weight | Reason |
|--------------|-------------|--------|
| krfp_1870 (contrib: positive) | 0.718 | Tricyclic nitrogen heteroarene. A rigid fused pi core with a positive gate is consistent with HOMO_avg above -5.5 and HOMO_LUMO_gap below 4.2, providing energy-level alignment rather than a polarity handle; its N-linked topology also supports the co-assembly role implied by HeteroSym = 1. Selected for the energy-alignment role, not for weight alone. |
| krfp_1153 (contrib: positive) | 0.635 | Tetra-attachment quaternary carbon. Acts as a steric node: it may tune Vol_ratio toward the 1.5 peak (contribution turns negative below 1.2) and support Rho_avg above 1.38, so it is retained for spatial complementarity rather than for frontier-orbital alignment; selected for the steric role, not for weight alone. |
| krfp_2511 (contrib: positive) | 0.588 | N-linked tricyclic nitrogen heteroarene. Contributes an electron-rich aromatic surface with an attachment topology distinct from krfp_1870, satisfying the heterogeneous-symmetry requirement (HeteroSym = 1, Mixture = 1) while avoiding a duplicate packing role. Selected for the symmetry role, not for weight alone. |
| krfp_4852 (contrib: positive) | 0.491 | Aryl thioether. Polarizable sulfur bridging an aromatic ring and the interface is consistent with the polarity window (TPSA_avg 60-68; Dipole_avg peaks at 1.8, 3.5 and 5.2) and with HOMO_avg above -5.5. |
| krfp_2854 (contrib: positive) | 0.487 | Dialkyl thioether. Sulfur supplies a hydrogen-bond acceptor that supports HBA_total above 8 and the interfacial dipole, but the aliphatic attachment adds sp3 flexibility, so it is recommended as a minor component rather than a backbone. |
| krfp_3881 (contrib: positive) | 0.425 | Aliphatic amine: raises local polarity through its nitrogen lone pair, with no pi contribution; a supporting polar handle for HBA_total above 8. |
| krfp_3891 (contrib: positive) | 0.289 | Tertiary aniline whose nitrogen supplies a hydrogen-bond acceptor for HBA_total above 8; a polar passivation handle with limited contribution to packing. |
| krfp_1641 (contrib: positive) | 0.286 | Disubstituted benzene acting as a rigid aromatic spacer; helps make an asymmetric/symmetric pair complementary rather than merely additive. |
| krfp_3223 (contrib: positive) | 0.264 | Flexible linker providing spatial complementarity (Vol_ratio near 1.5) and coverage. Its modest weight reflects the trade-off with sp2_total above 45, which favours conjugated rigidity over chain flexibility. |
| krfp_2778 (contrib: positive) | 0.199 | Supports anchoring interactions and volume balance (Vol_ratio near 1.5); retained as a low-weight supportive motif rather than a primary design handle. |

### Favorable Combinations

- **Combination**: Energy-alignment core
  - KRFP indices: krfp_1870, krfp_1153, krfp_4852
  - Reason: krfp_1870 and krfp_4852 may support favorable HOMO_avg and HOMO_LUMO_gap trends, while krfp_1153 supplies the steric node needed for the volume window.
  - Supporting evidence: HOMO_avg positive above -5.5; HOMO_LUMO_gap positive below 4.2; Vol_ratio near 1.5.

- **Combination**: Polarity/passivation group
  - KRFP indices: krfp_2511, krfp_2854, krfp_3881, krfp_3891
  - Reason: may provide HBA-rich or polar interface passivation and packing.
  - Supporting evidence: HBA_total above 8, Dipole_avg peaks at 1.8, 3.5 and 5.2, TPSA_avg best near 64 (60-68), Rho_avg above 1.38.

- **Combination**: Co-assembly/packing group
  - KRFP indices: krfp_1641, krfp_3223, krfp_2778
  - Reason: may support heterogeneous symmetry matching and balanced packing; all three fragments fall inside the twenty-five-fragment pool.
  - Supporting evidence: HeteroSym=1, Mixture=1, MACl_additive=1 and Vol_ratio peak near 1.5.

### Avoided Combinations

- **Combination**: Overloaded fused nitrogen pi without acceptor balance
  - KRFP indices: krfp_1870, krfp_2511, krfp_3127, krfp_2070
  - Reason: leaning on several fused nitrogen heteroarenes at once may raise sp2 and aromatic volume faster than it raises HBA_total, so the pi content may be met well before the polarity window is.
  - Supporting evidence: sp2_total positive above 45; HOMO_avg above -5.5; HBA_total above 8.

- **Combination**: Unbalanced acceptor clustering
  - KRFP indices: krfp_4256, krfp_4837, krfp_2641, krfp_2643
  - Reason: a carboxylic acid, a sulfonic acid and two amide/alcohol functions clustered together may push TPSA well beyond the 60-68 window, forming an insulating polar barrier rather than a passivating layer.
  - Supporting evidence: TPSA_avg best near 64 within 60-68; HBA_total above 8; Dipole_avg peaks at 1.8, 3.5 and 5.2.

- **Combination**: Flexible-linker cluster with insufficient rigidity
  - KRFP indices: krfp_3223, krfp_2854, krfp_3881
  - Reason: leaning on all three at once may add chain flexibility faster than polar benefit, lowering the sp2 fraction relative to the rigid packing that the volume and density windows require.
  - Supporting evidence: sp2_total positive above 45; Vol_ratio near 1.5 (positive above 1.2); Rho_avg above 1.38.

- **Combination**: Methoxy/ester-only polarity
  - KRFP indices: krfp_3999, krfp_1810, krfp_2778
  - Reason: two methoxyarenes and a phosphonate ester may raise HBA_total through ether and ester oxygens but contribute little strong hydrogen-bond donor character, so the polarity window is met numerically without the passivation it is meant to represent.
  - Supporting evidence: HBA_total above 8; TPSA_avg 60-68 (best 64); Dipole_avg peaks at 1.8, 3.5 and 5.2.
