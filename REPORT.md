# Cross-Layer Connections in Cross-Layer Transcoders: An Empirical Investigation

**Research Question**: How big a deal are the cross-layer connections in cross-layer transcoders? What are they really doing?

**Model**: Gemma Scope 2 CLTs on Gemma3-1B

---

## Background

Cross-Layer Transcoders (CLTs) are a key component of Anthropic's attribution graphs methodology. Unlike per-layer transcoders, CLTs organize features into L layers matching the model's structure, where each feature at layer ℓ can write to all subsequent MLP outputs (layers ℓ, ℓ+1, ..., L).

From the Anthropic methods paper:
> "An ℓth layer feature contributes to the reconstruction of the MLP outputs in layers ℓ, ℓ+1,…, L, using a separate set of linear decoder weights for each output layer."

This architecture enables **path collapsing** - representing multi-layer amplification chains as single features, reducing average path length from 3.7 to 2.3 in tested examples.

---

## Experiments

### Experiment 1: Cross-Layer Weight Analysis

**Goal**: Quantify how much decoder weight mass is in cross-layer vs same-layer connections.

![Weight Heatmap](figures/exp1_weight_heatmap.png)

**Key Findings**:
- Same-layer weight ratio: **19.46%**
- Cross-layer weight ratio: **80.54%**

The vast majority of the decoder weight mass lies in the cross-layer connections, suggesting they play a dominant role in the CLT's function.

![Weight Distribution](figures/exp1_weight_distribution.png)

---

### Experiment 2: Ablation Study

**Goal**: Measure the impact of removing cross-layer connections.

![FVU Comparison](figures/exp2_fvu_comparison.png)

**Key Findings**:
- Full CLT FVU: **28.65%**
- Ablated CLT FVU: **43.73%**
- FVU increase from ablation: **15.08%** (absolute) / **52.6%** (relative)

Removing cross-layer connections significantly degrades reconstruction quality, confirming that local features alone are insufficient to explain the activations at deeper layers.

![Delta Loss](figures/exp2_delta_loss.png)

---

### Experiment 3: Feature-Level Analysis

**Goal**: Identify which features rely most on cross-layer connections and verify they are active.

![Score Distribution](figures/exp3_score_distribution.png)

**Top Active Features**:
Scanning the `wikitext-2` dataset revealed highly active features with varying degrees of cross-layer reliance (Score = Cross/Local Norm Ratio).

| Layer | Feature | Max Act | Cross-Layer Score | Top Token |
|-------|---------|---------|-------------------|-----------|
| 20 | 40 | 6336.00 | 3.49 | `cons` |
| 20 | 985 | 6016.00 | 2.52 | `dem` |
| 21 | 132 | 5824.00 | 2.26 | `Bul` |
| 22 | 69 | 5632.00 | 2.05 | `unt` |
| 20 | 1904 | 5504.00 | 3.44 | `Mir` |

We see that highly active features often have significant cross-layer scores (>2.0), meaning they write more strongly to distant layers than to their own layer.

![Top Features by Layer](figures/exp3_top_features.png)

---

### Experiment 4: Layer Distance Analysis

**Goal**: Analyze how cross-layer connections vary with layer distance.

![Distance Decay](figures/exp4_distance_decay.png)

**Key Findings**:
- Decay pattern: The mean decoder norm decays as the distance between input and output layers increases.
- Decay factor (distance 0 to 5): **0.34** (Drop from 0.3783 to 0.1301)

While connections to nearby layers are strongest, there is a long tail of connections to distant layers that remains significant.

![Per-Layer Patterns](figures/exp4_per_layer_patterns.png)

![Detailed Heatmap](figures/exp4_detailed_heatmap.png)

---

### Experiment 5: The "Time Travel" Logit Lens

**Goal**: Determine if cross-layer writes predict the final token earlier than the layer they originate from.

**Key Findings**:
- **Average Logit Similarity**: **-0.003** (effectively zero)
- **Same Top Token Prediction**: **0/100** features

**Interpretation**: The vector written by a feature to a distant layer ($L_{out}$) typically predicts a completely different token than the vector written to the local layer ($L_{in}$). This suggests the feature **changes meaning** or contributes to a different semantic subspace as it propagates through the model, rather than simply amplifying a fixed concept.

---

### Experiment 6: Residual Stream Alignment

**Goal**: Quantify how much the cross-layer write mimics the actual transformation of the residual stream between layers ($\Delta R = R_{out} - R_{in}$).

**Key Findings**:
- **Global Average Alignment (Cosine Sim)**: **-0.0029**
- **Top Aligned Feature Similarity**: ~0.04 (very low)

**Interpretation**: The cross-layer connections do **not** simply mimic the aggregate function of the intermediate layers (the "shortcut" hypothesis). If they did, their write vectors would align with the actual change in the residual stream. Instead, they appear to be adding **orthogonal information** or performing a function that is distinct from the main residual path.

![Alignment Histogram](figures/exp6_alignment.png)

---

## Conclusions

1.  **Dominance of Cross-Layer Weights**: Cross-layer connections account for **~80%** of the decoder weight mass and are responsible for **~53%** of the reconstruction capability (FVU). They are not optional; they are central to the CLT's operation.
2.  **Distance Decay**: Connections are strongest locally but persist across the entire depth of the model.
3.  **Not Just Shortcuts**: Experiments 5 and 6 provide strong evidence against the "simple shortcut" hypothesis. Cross-layer connections do not merely "predict ahead" (Logit Lens) nor do they "approximate the path" (Residual Alignment).
4.  **Orthogonal Composition**: The results suggest that CLTs use cross-layer connections to **compose features orthogonally**. A feature at Layer X writes to Layer Y not to say "I am still here," but to say "I am a component of this *new* thing happening at Layer Y." This supports the view of CLTs as capturing **compositional algorithms** rather than just feature persistence.

### Implications for Circuit Analysis

When analyzing circuits using CLTs, researchers should **not** assume that a feature $f$ at $L_{in}$ means the same thing when it writes to $L_{out}$. The "time travel" experiment shows the semantic decoding changes completely. Attribution graphs must account for this transformation—the edge *is* the computation.

---

## References

1. [Attribution Graphs Methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) - Anthropic
2. [Gemma Scope 2 Blog](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/) - Google DeepMind
3. [Gemma Scope 2 1B PT on HuggingFace](https://huggingface.co/google/gemma-scope-2-1b-pt)