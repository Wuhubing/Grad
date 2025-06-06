# Multi-hop Knowledge Coupling Analysis Report

**Model:** gpt2  
**Gradient Dimension:** 32  
**Total Chains Analyzed:** 10  
**Analysis Time:** 61.45 seconds  

## Research Questions

- **RQ-1:** How are gradient similarities distributed between hops?
- **RQ-2:** Does GradSim predict ripple susceptibility?
- **RQ-3:** How do models/categories affect entanglement?

## Key Findings

### Overall Entanglement: 0.8244
- **Interpretation:** High knowledge coupling detected
- **Standard Deviation:** 0.1065
- **Range:** [0.6092, 0.9521]

### Local vs Distant Entanglement
- **Local Entanglement:** 0.9052
- **Distant Entanglement:** 0.4770
- **Ratio (Local/Distant):** 1.8977
- **Interpretation:** Local entanglement dominance

### Category-wise Analysis
1. **Location:** 0.8839 (n=1)
2. **Person:** 0.8197 (n=7)
3. **Time:** 0.8173 (n=1)
4. **Organization:** 0.8055 (n=1)

### Group Comparison
- **EntityFact:** 0.8252 (n=9)
- **EventRelation:** 0.8173 (n=1)

## Methodology

1. **Data Extraction:** Multi-hop chains from HotpotQA with 6-class categorization
2. **Gradient Computation:** MLP layer gradients with PCA reduction
3. **Similarity Analysis:** Cosine similarity between gradient vectors
4. **Entanglement Metrics:** Overall, local, and distant coupling measures

## Files Generated

- `chains/chains_train.jsonl` - Extracted reasoning chains
- `gradients/gradients.pkl` - Computed gradients with PCA
- `analysis/entanglement_results.csv` - Detailed results
- `analysis/aggregate_similarity_heatmap.png` - Similarity heatmap
- `analysis/entanglement_distribution.png` - Distribution plots
- `analysis/hop_entanglement_patterns.png` - Hop-wise patterns
- `analysis/summary_statistics.txt` - Detailed statistics

## Research Implications

**High Knowledge Coupling:** Strong parameter overlap suggests ripple effects likely.

---
*Report generated automatically by Knowledge Coupling Pipeline*
