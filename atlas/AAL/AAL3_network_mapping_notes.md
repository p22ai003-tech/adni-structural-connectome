# AAL3 (166-node) → network / system mapping

This file documents `AAL3_network_mapping.csv`, the lookup that groups the 166 AAL3 connectome nodes into
brain **networks** (functional) and **systems** (anatomical) for the dashboard's network-level analysis.

## Keying — read this first
Each row is keyed on **`matrix_idx` (1..166)**, the connectome matrix row index. It is built by sorting
`AAL3_labels.csv` by its `node` column and numbering the rows 1..166. This matches the canonical ordering
used by `connectome_labels_for_matrix(n)` and the `node` column in the live node tables
(`live_node_microstructure_long.csv`, `live_node_graph_long.csv`).

**Do not join on `node_name`.** In the live long tables `node_name` is a *contiguous* relabel
(`AAL_001..AAL_166`) and does **not** equal the AAL3 atlas value — the atlas has gaps at atlas values
35, 36, 81, 82. Example: matrix_idx 35 → atlas_value 37 → `Cingulate_Mid_L`. Always map via `matrix_idx`.

## Two schemes
- **`anatomical_system`** — exact AAL lobe/structure grouping (Frontal, Parietal, Temporal, Occipital,
  Limbic, BasalGanglia, Thalamus, Cerebellum, Brainstem). No approximation.
- **`functional_network`** — Yeo-7 resting-state networks for cortex (DMN, Limbic, Salience_VAN,
  DorsalAttention, Frontoparietal, Somatomotor, Visual) plus three non-cortical systems
  (Subcortical, Cerebellar, Brainstem) since Yeo-7 is cortex-only.

## Honest caveat on the functional scheme
AAL3 is an **anatomical** parcellation. Assigning anatomical parcels to **functional** (Yeo) networks is an
**approximation** — a region can straddle networks and published AAL→Yeo lookups disagree at the margins.
The structural connectomes here are also DWI tractography, so this asks "what is the structural backbone of
network X," not a functional-connectivity claim. Each assignment carries a one-line `rationale` in the CSV.

### Ambiguous calls (primary assignment → notable alternative)
- `Cingulate_Mid`, `ACC_*` → **Salience_VAN** (dorsal/mid cingulate is cingulo-opercular). Pregenual
  `ACC_pre` is sometimes placed in **DMN**.
- `Temporal_Sup` → **Salience_VAN** (TPJ / ventral attention). Posterior STG is alternatively **auditory**
  (folded into Somatomotor here for Heschl only).
- `Temporal_Inf` → **DMN** (lateral-temporal). Alternative: **Visual** (ventral stream).
- `Frontal_Sup_2` → **DorsalAttention** (FEF-adjacent). Alternative: **Frontoparietal**.
- `Hippocampus` / `ParaHippocampal` → **DMN** (medial-temporal subsystem); `Amygdala` → **Limbic**.
- `Heschl` → **Somatomotor** (Yeo lumps primary auditory with somatomotor).

## Low edge-support warning
Brainstem/neuromodulatory nuclei (VTA, SN, Red_N, LC, Raphe) and several thalamic nuclei are very small and
have sparse streamline coverage even in dense subjects. Network rows carry `n_nodes`/`n_edges`; treat
low-support networks (especially Brainstem) as exploratory.


## Functional networks — membership

- **Brainstem** (12 nodes): LC_L, LC_R, Raphe_D, Raphe_M, Red_N_L, Red_N_R, SN_pc_L, SN_pc_R, SN_pr_L, SN_pr_R, VTA_L, VTA_R
- **Cerebellar** (26 nodes): Cerebellum_10_L, Cerebellum_10_R, Cerebellum_3_L, Cerebellum_3_R, Cerebellum_4_5_L, Cerebellum_4_5_R, Cerebellum_6_L, Cerebellum_6_R, Cerebellum_7b_L, Cerebellum_7b_R, Cerebellum_8_L, Cerebellum_8_R, Cerebellum_9_L, Cerebellum_9_R, Cerebellum_Crus1_L, Cerebellum_Crus1_R, Cerebellum_Crus2_L, Cerebellum_Crus2_R, Vermis_10, Vermis_1_2, Vermis_3, Vermis_4_5, Vermis_6, Vermis_7, Vermis_8, Vermis_9
- **DMN** (20 nodes): Angular_L, Angular_R, Cingulate_Post_L, Cingulate_Post_R, Frontal_Med_Orb_L, Frontal_Med_Orb_R, Frontal_Sup_Medial_L, Frontal_Sup_Medial_R, Hippocampus_L, Hippocampus_R, ParaHippocampal_L, ParaHippocampal_R, Precuneus_L, Precuneus_R, Rectus_L, Rectus_R, Temporal_Inf_L, Temporal_Inf_R, Temporal_Mid_L, Temporal_Mid_R
- **DorsalAttention** (4 nodes): Frontal_Sup_2_L, Frontal_Sup_2_R, Parietal_Sup_L, Parietal_Sup_R
- **Frontoparietal** (8 nodes): Frontal_Inf_Oper_L, Frontal_Inf_Oper_R, Frontal_Inf_Tri_L, Frontal_Inf_Tri_R, Frontal_Mid_2_L, Frontal_Mid_2_R, Parietal_Inf_L, Parietal_Inf_R
- **Limbic** (18 nodes): Amygdala_L, Amygdala_R, Frontal_Inf_Orb_2_L, Frontal_Inf_Orb_2_R, OFCant_L, OFCant_R, OFClat_L, OFClat_R, OFCmed_L, OFCmed_R, OFCpost_L, OFCpost_R, Olfactory_L, Olfactory_R, Temporal_Pole_Mid_L, Temporal_Pole_Mid_R, Temporal_Pole_Sup_L, Temporal_Pole_Sup_R
- **Salience_VAN** (14 nodes): ACC_pre_L, ACC_pre_R, ACC_sub_L, ACC_sub_R, ACC_sup_L, ACC_sup_R, Cingulate_Mid_L, Cingulate_Mid_R, Insula_L, Insula_R, SupraMarginal_L, SupraMarginal_R, Temporal_Sup_L, Temporal_Sup_R
- **Somatomotor** (12 nodes): Heschl_L, Heschl_R, Paracentral_Lobule_L, Paracentral_Lobule_R, Postcentral_L, Postcentral_R, Precentral_L, Precentral_R, Rolandic_Oper_L, Rolandic_Oper_R, Supp_Motor_Area_L, Supp_Motor_Area_R
- **Subcortical** (38 nodes): Caudate_L, Caudate_R, N_Acc_L, N_Acc_R, Pallidum_L, Pallidum_R, Putamen_L, Putamen_R, Thal_AV_L, Thal_AV_R, Thal_IL_L, Thal_IL_R, Thal_LGN_L, Thal_LGN_R, Thal_LP_L, Thal_LP_R, Thal_MDl_L, Thal_MDl_R, Thal_MDm_L, Thal_MDm_R, Thal_MGN_L, Thal_MGN_R, Thal_PuA_L, Thal_PuA_R, Thal_PuI_L, Thal_PuI_R, Thal_PuL_L, Thal_PuL_R, Thal_PuM_L, Thal_PuM_R, Thal_Re_L, Thal_Re_R, Thal_VA_L, Thal_VA_R, Thal_VL_L, Thal_VL_R, Thal_VPL_L, Thal_VPL_R
- **Visual** (14 nodes): Calcarine_L, Calcarine_R, Cuneus_L, Cuneus_R, Fusiform_L, Fusiform_R, Lingual_L, Lingual_R, Occipital_Inf_L, Occipital_Inf_R, Occipital_Mid_L, Occipital_Mid_R, Occipital_Sup_L, Occipital_Sup_R

## Anatomical systems — membership

- **BasalGanglia** (8 nodes): Caudate_L, Caudate_R, N_Acc_L, N_Acc_R, Pallidum_L, Pallidum_R, Putamen_L, Putamen_R
- **Brainstem** (12 nodes): LC_L, LC_R, Raphe_D, Raphe_M, Red_N_L, Red_N_R, SN_pc_L, SN_pc_R, SN_pr_L, SN_pr_R, VTA_L, VTA_R
- **Cerebellum** (26 nodes): Cerebellum_10_L, Cerebellum_10_R, Cerebellum_3_L, Cerebellum_3_R, Cerebellum_4_5_L, Cerebellum_4_5_R, Cerebellum_6_L, Cerebellum_6_R, Cerebellum_7b_L, Cerebellum_7b_R, Cerebellum_8_L, Cerebellum_8_R, Cerebellum_9_L, Cerebellum_9_R, Cerebellum_Crus1_L, Cerebellum_Crus1_R, Cerebellum_Crus2_L, Cerebellum_Crus2_R, Vermis_10, Vermis_1_2, Vermis_3, Vermis_4_5, Vermis_6, Vermis_7, Vermis_8, Vermis_9
- **Frontal** (32 nodes): Frontal_Inf_Oper_L, Frontal_Inf_Oper_R, Frontal_Inf_Orb_2_L, Frontal_Inf_Orb_2_R, Frontal_Inf_Tri_L, Frontal_Inf_Tri_R, Frontal_Med_Orb_L, Frontal_Med_Orb_R, Frontal_Mid_2_L, Frontal_Mid_2_R, Frontal_Sup_2_L, Frontal_Sup_2_R, Frontal_Sup_Medial_L, Frontal_Sup_Medial_R, OFCant_L, OFCant_R, OFClat_L, OFClat_R, OFCmed_L, OFCmed_R, OFCpost_L, OFCpost_R, Olfactory_L, Olfactory_R, Precentral_L, Precentral_R, Rectus_L, Rectus_R, Rolandic_Oper_L, Rolandic_Oper_R, Supp_Motor_Area_L, Supp_Motor_Area_R
- **Limbic** (18 nodes): ACC_pre_L, ACC_pre_R, ACC_sub_L, ACC_sub_R, ACC_sup_L, ACC_sup_R, Amygdala_L, Amygdala_R, Cingulate_Mid_L, Cingulate_Mid_R, Cingulate_Post_L, Cingulate_Post_R, Hippocampus_L, Hippocampus_R, Insula_L, Insula_R, ParaHippocampal_L, ParaHippocampal_R
- **Occipital** (12 nodes): Calcarine_L, Calcarine_R, Cuneus_L, Cuneus_R, Lingual_L, Lingual_R, Occipital_Inf_L, Occipital_Inf_R, Occipital_Mid_L, Occipital_Mid_R, Occipital_Sup_L, Occipital_Sup_R
- **Parietal** (14 nodes): Angular_L, Angular_R, Paracentral_Lobule_L, Paracentral_Lobule_R, Parietal_Inf_L, Parietal_Inf_R, Parietal_Sup_L, Parietal_Sup_R, Postcentral_L, Postcentral_R, Precuneus_L, Precuneus_R, SupraMarginal_L, SupraMarginal_R
- **Temporal** (14 nodes): Fusiform_L, Fusiform_R, Heschl_L, Heschl_R, Temporal_Inf_L, Temporal_Inf_R, Temporal_Mid_L, Temporal_Mid_R, Temporal_Pole_Mid_L, Temporal_Pole_Mid_R, Temporal_Pole_Sup_L, Temporal_Pole_Sup_R, Temporal_Sup_L, Temporal_Sup_R
- **Thalamus** (30 nodes): Thal_AV_L, Thal_AV_R, Thal_IL_L, Thal_IL_R, Thal_LGN_L, Thal_LGN_R, Thal_LP_L, Thal_LP_R, Thal_MDl_L, Thal_MDl_R, Thal_MDm_L, Thal_MDm_R, Thal_MGN_L, Thal_MGN_R, Thal_PuA_L, Thal_PuA_R, Thal_PuI_L, Thal_PuI_R, Thal_PuL_L, Thal_PuL_R, Thal_PuM_L, Thal_PuM_R, Thal_Re_L, Thal_Re_R, Thal_VA_L, Thal_VA_R, Thal_VL_L, Thal_VL_R, Thal_VPL_L, Thal_VPL_R
