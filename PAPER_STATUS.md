# Paper Status

Status of the Transcender paper package and the `transcender-mlx` MLX release repo as of 2026-03-24.

---

## Canonical State

- **Canonical source of truth:** [`paper/main.tex`](paper/main.tex) is the release-audited paper narrative. Its tables, claims, artifact references, and reproduction commands reflect the final empirical state.
- **Track A:** GPT-OSS 20B N=15 follow-up is complete, and the Qwen3-30B-A3B cross-model sparse-MoE reproduction is complete.
- **Track B:** The naive Gemma-to-GPT-OSS cascade remains a scoped negative baseline for the tested configuration.
- **Track C:** The dense follow-up is complete. Compute-both quality recovery is established, but no practical real selective-depth frontier was found on the tested dense families in this runtime.
- **Repo docs:** Release-facing docs in `transcender-mlx` align to the narrow public claim: a viable penultimate-layer selective-exit frontier appears on both tested sparse MoE families; operating-point quality is model-dependent; dense models did not show a practical real selective-depth frontier on this runtime.
- **GPU validation tooling:** `scripts/track_a_gpu/` provides a manual-reference off-MLX diagnostic path for Track A. It is useful for external-validity checks, but it is not part of the paper's canonical MLX evidence base. NVIDIA H200 structural reproductions confirm the penultimate-layer frontier on both MoE families: GPT-OSS 20B (bf16) L21=0.808 L22=0.879; Qwen3-30B-A3B (fp16) L45=0.832 L46=0.916. Cliff probes confirm monotonic degradation one layer further back (GPT-OSS L20=0.589; Qwen3 L44=0.793). Layer ordering survives extension from 48 to 64 generated tokens. Among tested oracle-style acceptance diagnostics, plain penultimate `top1_agree` remains the strongest simple rule; added gating did not improve the picture. GPU-side karma logistic classifiers (precision ~0.86) materially outperform entropy-only baselines (precision ~0.53–0.56). Frontier structure is robust off-MLX; cliff severity is runtime-sensitive. All GPU results are verifier-path diagnostics under shared full-depth context.
- **Markdown whitepapers:** Older markdown narratives are retained, but they are labeled as superseded or historical rather than presented as current evidence.
- **Bibliography / citation / license:** `paper/references.bib`, `CITATION.cff`, and the MIT license are coherent for public release. The author metadata remains Beomjune Son.

---

## Remaining Human Action Before arXiv

### 1. `.bbl` file generation / PDF Compilation

- **Completed:** The paper has been successfully built to `transcender.pdf` (which implicitly means the bibliography is resolved). The final `transcender.pdf` is in the repository root.
- **Action:** Include `transcender.pdf` (or its source `main.tex` and `.bbl`) in the arXiv upload bundle.

### 2. Figures

- `paper/main.tex` already includes `paper/figures/quality_cliff.png` and `paper/figures/category_breakdown.png`.
- **Action:** Include those referenced PNGs in the arXiv upload bundle. Extra local figures not referenced by `main.tex` can stay out of the upload.

### 3. Affiliation (optional)

- `main.tex` currently has no affiliation line in the author block.
- **Action:** Add affiliation only if desired before submission.

---

## What Should Remain Fixed

- **Measured numbers** in JSON artifacts, the LaTeX paper, and the aligned release docs
- **Track A framing:** two tested sparse MoE families (GPT-OSS 20B and Qwen3-30B-A3B), with a viable penultimate-layer frontier on both and model-dependent quality
- **`avg_layers_saved` interpretation:** mean skipped layers per generated token, not a casual compute-savings percentage
- **Track B framing:** scoped negative baseline for one naive cascade implementation, one model pair, and one runtime
- **Track C framing:** compute-both recovery is real; practical dense selective-depth remains negative in the tested runtime
- **Thesis statement:** cross-layer composition control is the core problem, not generic early exit in the abstract

---

## arXiv Upload Checklist

1. Verify the final `README.md` and `docs/` snapshots are consistent with `paper/main.tex`
2. Compile to generate `paper/references.bbl`
3. Bundle the figures explicitly referenced by `paper/main.tex`
4. Create the arXiv tarball with `main.tex`, `references.bbl`, and any explicitly included figures
5. Upload to arXiv as the chosen category
6. Add the public `transcender-mlx` GitHub repository URL to the arXiv comments field after release
