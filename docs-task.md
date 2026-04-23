# Session 27 — 2026-04-23 — Review Minibook Generation

## What we covered
- Discussed requirements for a review minibook covering all material from sessions 1–26
- Clarified format (PDF-ready LaTeX), depth (concepts + pseudocode), questions (a few per chapter), organization (by phase, matching PLAN.md), file location (`docs/`), and build pipeline (Makefile)
- Read all relevant source files and PROGRESS.md to ensure accuracy
- Created a 7-chapter LaTeX minibook covering Phases 1–7
- Split the initial single-file LaTeX into modular structure (one file per chapter)
- Diagnosed and fixed all compilation errors (algorithm2e inside tcolorbox incompatibility)

## Key learnings
- The `algorithm2e` package's `\begin{algorithm2e}[H]` environment cannot be used inside `tcolorbox` environments because it uses `\@float` internally, which conflicts with tcolorbox's savebox mechanism
- Fix: replace `algorithm2e` blocks with plain formatted pseudocode using `\\` line breaks, `\textbf{}` keywords, and `\hfill\textrm{\small ...}` for right-aligned comments
- `\texorpdfstring` is needed for hyperref compatibility when section titles contain math (`$O(n^2)$`) or special formatting (`GQA`)
- LaTeX font warning `OMS/cmtt/m/n` undefined is cosmetic — bold typewriter isn't available in that font encoding, falls back to regular weight
- `pdflatex` must be run twice for correct table of contents and cross-references

## Code written
- `docs/llm_review_guide.tex` — Main LaTeX file with preamble (packages, colors, tcolorbox styles, title) and `\input{}` calls for each chapter
- `docs/chapters/chapter1_tokenization.tex` — Tokenization spectrum, SimpleTokenizer, BPE (training + encoding pseudocode), sliding-window dataset
- `docs/chapters/chapter2_attention.tex` — Simplified self-attention, scaled dot-product, causal masking (TikZ diagram), multi-head attention, soft dictionary, O(n²) complexity
- `docs/chapters/chapter3_gpt_architecture.tex` — Token+positional embeddings, LayerNorm, residual connections, pre-norm vs post-norm, GELU FFN, TransformerBlock (TikZ diagram), GPT-2 architecture, generation strategies
- `docs/chapters/chapter4_pretraining.tex` — Cross-entropy loss, AdamW, warmup+cosine decay LR schedule, weight initialization, gradient accumulation, mixed precision
- `docs/chapters/chapter5_pretrained_weights.tex` — HuggingFace weight layout, Conv1D transpose, weight mapping table, weight tying, perplexity evaluation
- `docs/chapters/chapter6_finetuning.tex` — Classification head, instruction fine-tuning with masked loss, attention masks, length-grouped batching, gradient checkpointing, LoRA
- `docs/chapters/chapter7_modern_architecture.tex` — RMSNorm, RoPE (2D rotation + rotate-and-interleave), SwiGLU, GQA (TikZ diagram + comparison table)
- `docs/Makefile` — Build target that runs pdflatex twice, clean target

## PLAN.md items completed
- (No PLAN.md items — this was a review/documentation task, not a new implementation phase)

## Open questions
- The minibook currently covers Phases 1–7 (sessions 1–26). As new phases are implemented (KV Cache, Flash Attention, DeepSeek, RLHF/DPO), corresponding chapters should be added to `docs/chapters/`
- Two minor overfull hbox warnings remain (long monospace paths in chapter 6 and chapter 7) — these are cosmetic and don't affect readability
- The `OMS/cmtt/m/n` font shape warning is cosmetic (bold typewriter unavailable) and can be ignored

## File structure
```
docs/
├── Makefile                              # make / make clean
├── llm_review_guide.tex                  # Preamble + \input{} calls
├── llm_review_guide.pdf                  # Compiled 29-page PDF
└── chapters/
    ├── chapter1_tokenization.tex
    ├── chapter2_attention.tex
    ├── chapter3_gpt_architecture.tex
    ├── chapter4_pretraining.tex
    ├── chapter5_pretrained_weights.tex
    ├── chapter6_finetuning.tex
    └── chapter7_modern_architecture.tex
```

## Build instructions
```bash
cd docs && make        # builds llm_review_guide.pdf (runs pdflatex twice)
cd docs && make clean  # removes .pdf, .aux, .log, .out, .toc, etc.
```

Requires a TeX Live installation with: `amsmath`, `amssymb`, `booktabs`, `enumitem`, `hyperref`, `tikz` (with libraries: positioning, arrows.meta, fit, calc, decorations.pathreplacing), `xcolor`, `tcolorbox` (with libraries: skins, breakable), `alltt`.

## Error history and fixes
1. **`algorithm2e` inside `tcolorbox`**: `\begin{algorithm2e}[H]` environments cannot be nested inside `tcolorbox` environments. The `pseudocode` tcolorbox wraps content in a savebox that conflicts with algorithm2e's float mechanism. **Fix**: replaced all algorithm2e blocks with plain formatted pseudocode using line breaks (`\\`) and bold keywords (`\textbf{for}`, `\textbf{if}`, etc.). Removed `\usepackage{algorithm2e}`, added `\usepackage{alltt}` (though ultimately not used directly — the pseudocode is now just raw text inside tcolorboxes).
2. **`\DontPrintSemicolons`, `\KwIn`, `\KwOut`, `\With`, `\Comment`, `\tcp*`**: These were all `algorithm2e` commands that became undefined after removing the package. Eliminated by the pseudocode rewrite.
3. **`\texorpdfstring` for hyperref**: Section titles containing `$O(n^2)$` and `GQA` caused hyperref warnings about math tokens in PDF strings. **Fix**: wrapped with `\texorpdfstring{$O(n^2)$}{O(n²)}` and `\texorpdfstring{GQA}{GQA}`.
4. **Overfull hbox in chapter 6**: The long `\texttt{torch.utils.checkpoint.checkpoint(block, out, attn\_mask, use\_reentrant=False)}` text overflowed. **Fix**: shortened to `\texttt{torch.utils.checkpoint.checkpoint()}` with `use\_reentrant=False` described in prose.
5. **Font warning `OMS/cmtt/m/n` undefined**: Bold typewriter font isn't available in OT1 encoding. **Decision**: ignored — cosmetic only, falls back to regular weight monospace.