# Paper Status

Status of the paper package as of 2026-03-22.

---

## What Is Ready

- **LaTeX draft** (`paper/main.tex`): Complete. Aligned with the locked final research position across all three tracks. All tables, equations, and prose reflect measured evidence. Reproduction commands and artifact references updated to the cleaned repository structure.
- **Bibliography** (`paper/references.bib`): Three entries. Covers the two primary related-work citations (Leviathan et al. speculative decoding, Elbayad et al. depth-adaptive) and the Gemma report.
- **Whitepaper narrative** (`paper/Transcender_Final_Whitepaper_v2.md`): Evidence-audited. Consistent with `main.tex`.
- **Benchmark artifacts**: All JSON files present in `artifacts/`. No artifact has been modified from its original benchmark output.

---

## What Requires Human Action Before arXiv

### 1. Author Metadata
- `main.tex` currently says: `\author{Repository Draft \\ Author metadata to be added before submission}`
- **Action:** Replace with real author name(s) and affiliation(s).

### 2. Bibliography Verification
- `references.bib` entry for Elbayad et al. has: `note = {Placeholder venue metadata; verify before submission.}`
- `references.bib` entry for Gemma has: `note = {Author formatting may need normalization before submission.}`
- **Action:** Verify venue, year, and author formatting for all three entries before submission.

### 3. .bbl File Generation
- arXiv requires either a `.bbl` file or inline `\thebibliography`. No local TeX engine was available to compile.
- **Action:** Compile `paper/main.tex` with `pdflatex` + `bibtex` (or on Overleaf) to generate `paper/references.bbl`. Include the `.bbl` in the arXiv upload.

### 4. Figures
- `paper/figures/` contains 10 PNGs. None are currently referenced by `\includegraphics` in `main.tex`. The paper uses tables only.
- **Action:** Decide whether to include any figures in the arXiv submission. If yes, add `\includegraphics` commands to `main.tex`. If no, the figures directory can be excluded from the arXiv tarball.

### 5. License
- `LICENSE` file exists at repo root with MIT license.
- **Action:** Verify this is the intended license before public release.

### 6. CITATION.cff
- `CITATION.cff` exists at repo root with placeholder author metadata.
- **Action:** Update with real author name(s), ORCID(s), and finalize the DOI field after publication.

---

## What Should Not Be Changed

- **Measured numbers** in any JSON artifact, LaTeX table, or documentation
- **Track A canonical result** (0.969 exact match, ~49.5% layers saved)
- **Track B scoped positioning** (negative comparison baseline, not a general cascade claim)
- **Track C dense limitation framing** (benchmark-side limitation generalizes; KL profile is family-sensitive)
- **Thesis statement** consistency across all documents

---

## arXiv Upload Checklist

1. Update author metadata in `main.tex`
2. Verify `references.bib` entries
3. Compile to generate `.bbl` file
4. Decide on figures inclusion
5. Create tarball: `main.tex`, `references.bbl`, (optionally `figures/`)
6. Upload to arXiv as `cs.CL` or `cs.LG`
7. Add GitHub repository URL to arXiv comments field after repo is public
