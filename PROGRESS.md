# Progress Log

<!-- Session entries are added by the agent at the end of each working session. -->

## Session 1 — 2026-03-19 — Tokenization (Part 1)

### What we covered
- Discussed the tokenization spectrum: character-level → word-level → subword (BPE)
- Explained the regex pattern used for splitting text into tokens
- Built a `SimpleTokenizer` class with encode/decode functionality
- Added special tokens (`<|unk|>` and `<|endoftext|>`) for handling unknown tokens and document boundaries
- Discussed BPE algorithm conceptually with a worked example

### Key learnings
- Tokenization is the bridge between raw text and the integer sequences an LLM operates on
- `re.split()` with a capturing group preserves delimiters in the output
- Word-level tokenizers suffer from closed vocabulary, vocabulary explosion, and lack of morphological awareness — motivating subword approaches like BPE
- BPE iteratively merges the most frequent adjacent pair, building up from characters to subwords

### Code written
- `src/llm_from_scratch/tokens.py` — `SimpleTokenizer` class with `_tokenize`, `encode`, `decode`, special token support
- `examples/tokens.py` — Example script demonstrating the tokenizer

### PLAN.md items completed
- [x] Understand the tokenization spectrum: character → word → subword (BPE)
- [x] Build a simple tokenizer from scratch (vocabulary building, encode/decode)

### Open questions
- BPE implementation is next — `BPETokenizer` class with training (merge loop) and encoding (replay merges)
