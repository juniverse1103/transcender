# Paper Status

Status of the paper package as of 2026-03-23.

---

## Canonical State

- **Canonical source of truth:** [`paper/main.tex`](paper/main.tex) is the release-audited paper narrative. Its tables, claims, artifact references, and reproduction commands reflect the final empirical state.
- **Track A:** GPT-OSS 20B N=15 follow-up is complete, and the Qwen3-30B-A3B cross-model sparse-MoE reproduction is complete.
- **Track B:** The naive Gemma-to-GPT-OSS cascade remains a scoped negative baseline for the tested configuration.
- **Track C:** The dense follow-up is complete. Compute-both quality recovery is established, but no practical real selective-depth frontier was found on the tested dense families in this runtime.
- **Repo docs:** Release-facing docs align to the narrow public claim: a viable penultimate-layer selective-exit frontier appears on both tested sparse MoE families; operating-point quality is model-dependent; dense models did not show a practical real selective-depth frontier on this runtime.
- **Markdown whitepapers:** Older markdown narratives are retained, but they are labeled as superseded or historical rather than presented as current evidence.
- **Bibliography / citation / license:** `paper/references.bib`, `CITATION.cff`, and the MIT license are coherent for public release. The author metadata remains Beomjune Son.

---

## Remaining Human Action Before arXiv

### 1. `.bbl` file generation

- arXiv requires either a `.bbl` file or inline `\thebibliography`. No local TeX engine was available to compile in this workspace.
- **Action:** Compile `paper/main.tex` with `pdflatex` + `bibtex` (or on Overleaf) to generate `paper/references.bbl`, then include the `.bbl` in the arXiv upload.

### 2. Figures

- `paper/figures/` contains PNG assets, but `main.tex` currently relies on tables rather than `\includegraphics`.
- **Action:** Decide whether to include figures in the arXiv submission. If yes, add them explicitly to `main.tex`; if no, the figures directory can be omitted from the upload bundle.

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
3. Decide on figure inclusion
4. Create the arXiv tarball with `main.tex`, `references.bbl`, and any explicitly included figures
5. Upload to arXiv as the chosen category
6. Add the public GitHub repository URL to the arXiv comments field after release
