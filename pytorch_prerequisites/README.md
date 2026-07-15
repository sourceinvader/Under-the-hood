# PyTorch Prerequisites for *Under the Hood*

This guide maps out exactly how much PyTorch you need to know to work through all 35 projects in this repository, and where each piece of knowledge first becomes necessary. It was produced by reading every project's `README.md`, `build.py`, and `step_*.py` files and cataloging the concrete PyTorch APIs used versus the concepts each project assumes you already have — every formula and API reference below and in the linked files is checked directly against this repo's actual code, not recalled from the general literature.

The book's own stated bar (`README.md`) is: *"You should be comfortable with Python and have seen a tensor before. Everything else is built up from scratch."* That's true in spirit, but "everything else" undersells it a little in the back half of the book — projects 9–35 lean on a fair amount of PyTorch idiom (hooks, `named_modules()`, distributed collectives, HuggingFace-style model conventions) that the book doesn't stop to teach because it isn't really about PyTorch at that point — it's about systems, algorithms, and math, and PyTorch is just the notation. This guide is the missing "notation primer."

---

## How this guide is organized

| File | Covers | Needed by |
|---|---|---|
| [`01_math-foundations.ipynb`](01_math-foundations.ipynb) | The non-PyTorch math underneath everything else: matrix multiplication, the chain rule, exponentials/logarithms and numerical stability, probability and negative log-likelihood, softmax/cross-entropy, norms. Read this first if your math is rusty; every other file links back to it instead of re-deriving formulas. | Before Project 2 |
| [`02_tensors-and-autograd.ipynb`](02_tensors-and-autograd.ipynb) | **Tier 1 — Core.** Tensors, autograd mechanics (with a worked chain-rule example and the math behind gradient accumulation and in-place-op hazards), broadcasting/indexing, shape manipulation and strides, `nn.Module` basics, the functional API's actual formulas, the AdamW update equations, gradient clipping. | Project 2 onward |
| [`03_systems-and-training.ipynb`](03_systems-and-training.ipynb) | **Tier 2 — Intermediate.** Module introspection, checkpointing, hooks, the numerical case for mixed precision and loss scaling, CUDA/device management, the scaled dot-product attention formula and why the `1/√d` scaling exists, and the three `torch.distributed` collectives (all-reduce, all-gather, reduce-scatter) with FSDP. | Roughly Projects 5–17 |
| [`04_advanced-topics.ipynb`](04_advanced-topics.ipynb) | **Tier 3 — Advanced.** The actual formulas behind MoE routing, LoRA, all four preference-optimization losses (DPO/KTO/ORPO/SimPO), the GRPO policy gradient, quantization, patch embeddings, Mamba/RWKV's recurrences, RoPE extension methods (PI/NTK/YaRN), and InfoNCE — each transcribed directly from this repo's `build.py`. | Projects 18–35 |

Read them in order if you're starting cold; jump straight to the relevant one if you already know the basics and just want the math for a specific later project.

---

## TL;DR

- **To start Project 1:** no PyTorch at all. It's pure Python.
- **To start Project 2** (first project that imports `torch`): you need [`02_tensors-and-autograd.ipynb`](02_tensors-and-autograd.ipynb) — tensors, autograd basics, broadcasting, indexing, and eventually `nn.Module`.
- **To comfortably reach Project 17** (end of the systems/inference arc): add [`03_systems-and-training.ipynb`](03_systems-and-training.ipynb) — module introspection, hooks, mixed precision, device/CUDA management, and `torch.distributed`/FSDP.
- **To get through Projects 18–35** without constantly stopping to look things up: add [`04_advanced-topics.ipynb`](04_advanced-topics.ipynb) — the pattern here is less "new PyTorch API" and more "the code becomes thinner and more like transcribed pseudocode," so the prerequisite shifts from *tensor mechanics* to *reading partially-specified code and filling in the undefined pieces yourself*.
- If you already know research-level PyTorch (you've trained a transformer from scratch before, you know what `requires_grad` and `register_buffer` do, you've used `torch.distributed` at least once), you can start on Project 1 today and treat this guide as a reference rather than a study plan.

---

## What you do *not* need to know beforehand

The book's whole premise is building things from scratch, and it delivers on that for the core algorithmic content. You do **not** need prior knowledge of:

| Concept | Built from scratch in |
|---|---|
| Reverse-mode automatic differentiation / backpropagation | Project 1 (a pure-Python scalar autograd engine, no PyTorch) |
| Byte-Pair Encoding tokenization | Project 3 (pure Python, no PyTorch) |
| Scaled dot-product attention, causal masking, multi-head split | Project 4 (raw tensors, no `nn.Module`) |
| The GPT architecture itself (embeddings, blocks, LM head, weight tying) | Project 5 |
| LayerNorm / RMSNorm math | Project 7 (hand-derived before the built-in is ever used) |
| RoPE, and later YaRN / NTK-aware scaling | Projects 7 and 16 |
| FlashAttention's tiling / online-softmax algorithm | Project 8 (CPU reference implementation) |
| The KV cache | Project 13 |
| Mixture-of-Experts routing | Project 18 |
| DPO / KTO / ORPO / SimPO loss derivations | Project 24 |
| Quantization math (scale/zero-point, symmetric int8/int4) | Project 27 |
| Mamba-style selective state-space models and RWKV | Project 30 |

If you already know some of these, great — the book will feel like confirmation rather than discovery in those chapters. If you don't, that's the point: you're not expected to. (If you'd still like the mathematical basis for any of these ahead of time, it's in [`04_advanced-topics.ipynb`](04_advanced-topics.ipynb) and [`01_math-foundations.ipynb`](01_math-foundations.ipynb) — this guide explains the underlying math without doing the book's job of walking you through building the code.)

---

## Ecosystem libraries you'll also need

These aren't PyTorch, but the book leans on them alongside it:

| Library | Where it's needed |
|---|---|
| `numpy` | Throughout; the sole library in Projects 10 and 19 |
| `matplotlib` | Loss curves, sweep plots |
| `tiktoken` | Comparing the from-scratch tokenizer (Project 3) against a production one |
| `pandas` | Scaling-law curve fitting (Project 19) |
| `datasets`, `huggingface_hub` | Real dataset access (Projects 9–10, 24) |
| `transformers` (HuggingFace) | Several later chapters (22, 24, 25, 31) quietly switch to HF-style model conventions — `.logits`, `.last_hidden_state`, `.generate(...)`, `output_hidden_states=True` / `.hidden_states`, `model.transformer.h` — instead of the book's own from-scratch GPT attribute names. Worth knowing this convention exists as its own small prerequisite; the repo doesn't call it out when it happens. |
| `faiss-cpu` | Vector search (Project 28) |
| `mmh3` | MinHash deduplication (Project 10) |
| `bitsandbytes`, `flash-attn`, `llama-cpp-python` | Per-project extras for a handful of chapters; installed only if you do that project (see `setup/02_installing-dependencies.md`) |

---

## Per-project quick reference

A rough density rating for "how much genuinely new PyTorch mechanics does this project exercise" — **None** (no `torch` import at all), **Light**, **Moderate**, or **Heavy**. Density is not the same as difficulty: several "None"/"Light" projects (19, 20, 23, 26, 34) are conceptually substantial but happen to express that substance in prose, math, or plain Python rather than tensor code.

| # | Project | Torch density | What's new here |
|---|---|---|---|
| 1 | The Learning Machine | **None** | Pure Python — builds a scalar autograd engine from scratch |
| 2 | Predicting the Next Character | Light–Moderate | First `torch` usage: raw tensors, manual `requires_grad`/`.backward()`, hand-written SGD, `F.cross_entropy` |
| 3 | Building a Tokenizer | **None** | Pure Python BPE |
| 4 | Attention from Scratch | Light–Moderate | Raw-tensor Q/K/V, `masked_fill`+softmax, batched matmul — no `nn.Module`, no training |
| 5 | Your GPT from a Blank File | **Heavy** | Full `nn.Module` GPT: `register_buffer`, weight tying, `nn.Embedding`/`Linear`/`LayerNorm`, `AdamW`, generation loop |
| 6 | From Prototype to nanoGPT | Moderate | Optimizer parameter groups, `named_modules()`/`modules()`, scaled init |
| 7 | The Details That Matter | Moderate | Custom `nn.Module` norm/activation layers (RMSNorm, SwiGLU) |
| 8 | Flash Attention and Tiled Kernels | Moderate | Tiled/online-softmax CPU reference — forward-only, no autograd anywhere |
| 9 | Pretraining on the Real Web | **None** | Data-engineering prose (shards, mmap, packing) — no executable torch code |
| 10 | Data Curation and Contamination | **None** | NumPy + `mmh3` MinHash dedup |
| 11 | Training Debugging: Spikes, NaNs, Profiling | Moderate–Heavy | Forward hooks, non-recursive parameter iteration, `GradScaler` |
| 12 | Distributed Training: FSDP and ZeRO | **Heavy** | `torch.distributed`, `mp.spawn`, hand-rolled all-gather/reduce-scatter, real `FullyShardedDataParallel` |
| 13 | Fast Inference: The KV Cache | Light–Moderate | Cache tensor indexing/bookkeeping across autoregressive steps |
| 14 | Speculative Decoding | Moderate | `torch.multinomial`, `.gather()`, rejection sampling |
| 15 | Grouped Query Attention | Moderate | `expand()`+`reshape()` KV-head repeat trick |
| 16 | Long-Context Extension (RoPE/YaRN/NTK) | Light | Math-heavy, thin API — mostly scalar/frequency arithmetic |
| 17 | Continuous Batching and PagedAttention | Moderate | Preallocated tensor "arena" indexed like OS memory pages; naive reference kernel |
| 18 | Mixture of Experts | Moderate | `torch.topk`, `scatter_`, `bincount` |
| 19 | Scaling Laws | **None** | NumPy/pandas log-log curve fitting over a pre-existing CSV |
| 20 | Autonomous Experimentation | **None** | Agent-orchestration policy logic, no ML library |
| 21 | Fine-Tuning and Instruction Tuning (LoRA) | Light | One self-contained `LoRALinear` class |
| 22 | Evaluation Methodology | Light–Moderate | `log_softmax`+`gather` idiom; first appearance of HF-style `.generate()`/`.logits` |
| 23 | Reward Models and RLHF (GRPO) | Light / pseudocode | Reward head + REINFORCE-style loss; most undefined helpers of any project |
| 24 | DPO and Preference Optimization | Moderate–Heavy | `F.logsigmoid`, masked log-probs, frozen reference model alongside trainable policy |
| 25 | Test-Time Reasoning (CoT, Self-Consistency, Best-of-N) | Light | One `ORM` class; rest is search/voting orchestration |
| 26 | Tool Use and Function Calling | **None** | JSON schemas + regex-based dispatch loop |
| 27 | Quantization and Deployment | Moderate–Heavy | int8/int4 round-trip, `.weight.data`, module traversal — zero autograd |
| 28 | Retrieval-Augmented Generation | Light | One `info_nce_loss` function; rest is FAISS/NumPy |
| 29 | Multimodal: A Tiny Vision-Language Model | Moderate | `nn.Conv2d` patch embedding, token concatenation, masked cross-entropy |
| 30 | Non-Transformer Architectures (Mamba, RWKV) | **Heavy** | Dense broadcasting, manual sequential recurrence, log-space stability tricks |
| 31 | Layer Freezing and Transfer | Light–Moderate | `requires_grad` sweeps, hand-written CKA, HF-style model attributes |
| 32 | Fusing Independently Trained Specialists | Light | One router class, manual soft-label loss |
| 33 | The Interface Specification | Light | Attribute/type introspection, `isnan`/RMS runtime probes |
| 34 | Incremental Assembly | **None** | Orchestration pseudocode |
| 35 | Your Architecture | Near-zero | Two torch calls (`no_grad`, implicit `backward`) in an open-ended capstone |

---

## Repo-specific reading notes

These aren't PyTorch prerequisites so much as things worth knowing about *this repository* before you assume its code is complete, runnable, or internally consistent:

- **`build.py` is generated, not hand-written.** `tools/extract_code.py` mechanically concatenates the fenced code blocks from the book's own chapter markdown. That's why some `build.py` files (9, 19, 20, 23, 26, 34, 35 especially) read as fragments referencing undefined names — they're pedagogical excerpts, not standalone programs. Where a step is prose-only in the book, there's no corresponding `step_*.py` file at all.
- **Several `README.md` files reference a `break_it.py` that doesn't exist on disk** for that project (confirmed missing for at least Projects 30, 31, 32, 34, 35) — the BREAK IT content for those exists only in the book's prose.
- **`nn.RMSNorm` (Project 30) needs PyTorch ≥ 2.4**, but `requirements.txt`/`pyproject.toml` only pin `torch>=2.2,<3.0`. Worth upgrading before that project specifically.
- **The book quietly switches conventions partway through.** Projects 1–17 use the repo's own from-scratch `GPT`/`Block` classes (attributes like `token_embedding`, `blocks`). Projects 22, 24, 25, and especially 31 assume HuggingFace `transformers`-style objects instead (`.logits`, `.generate()`, `model.transformer.h`, `output_hidden_states=True`) — despite `transformers` never being imported or declared as a dependency anywhere in the repo. If you try to run these chapters' code verbatim against your own Project 5 GPT class, expect `AttributeError`s until you either adapt the code or swap in an actual HF model.
- **Project 12's `from nanochat.model import GPT, TransformerBlock` is aspirational**, not runnable as-is — there's no `nanochat` package in this repo; you're expected to supply your own prior-project model.
- **Project 29's `ViTBlock`** is used but never defined anywhere in the repo — you're expected to write a standard pre-norm transformer encoder block yourself, by analogy to the causal GPT block from Projects 4–5 (minus the causal mask).
- **`build.py` in Project 5 calls `model.eval()` once before generation but never `model.train()` again**, and validation-loss estimation uses only `torch.no_grad()` rather than the `model.eval()`/`model.train()` pairing shown in the book's own `step_12` file. It's numerically silent in this repo because every default run uses `dropout=0.0` — but it's a real bug pattern to notice rather than copy.

---

## Suggested prep path

- **Never used PyTorch:** work through the official 60-minute blitz plus one "train a small classifier" tutorial before starting Project 2, then read [`01_math-foundations.ipynb`](01_math-foundations.ipynb) and [`02_tensors-and-autograd.ipynb`](02_tensors-and-autograd.ipynb) in full. That covers essentially all of Tier 1. Come back to [`03_systems-and-training.ipynb`](03_systems-and-training.ipynb) organically when you reach Projects 11–12.
- **Comfortable with basic PyTorch (tensors, autograd, a simple `nn.Module`), but never touched distributed training, hooks, or quantization:** you can start immediately. Skim [`03_systems-and-training.ipynb`](03_systems-and-training.ipynb) before Project 11 and [`04_advanced-topics.ipynb`](04_advanced-topics.ipynb)'s quantization/state-space sections before Projects 27 and 30 so the version trap and the "always dequantizes to float" caveat don't surprise you mid-chapter.
- **Already comfortable at a research-engineering level:** start on Project 1 today and use this guide as a lookup table (via the per-project table above, and the math in [`04_advanced-topics.ipynb`](04_advanced-topics.ipynb) for the specific later chapters) rather than a study plan — the "Repo-specific reading notes" section above is probably the most useful part for you, since it flags the places where the repo's code diverges from what the book's prose promises.

---

## Glossary

Acronyms used in the tables and notes above, expanded once so you don't have to guess — the linked files each have their own glossary for concepts, not just acronyms:

- **MoE** — Mixture of Experts (Project 18)
- **LoRA** — Low-Rank Adaptation (Project 21)
- **RLHF** — Reinforcement Learning from Human Feedback (Project 23)
- **GRPO** — Group Relative Policy Optimization, the specific RLHF algorithm Project 23 implements
- **DPO / KTO / ORPO / SimPO** — Direct Preference Optimization and three later variants, all covered in Project 24
- **RAG** — Retrieval-Augmented Generation (Project 28)
- **CKA** — Centered Kernel Alignment, a similarity measure between two models' hidden states (Project 31)
- **FSDP** — Fully Sharded Data Parallel, PyTorch's built-in sharded-training implementation (Project 12)
- **KV cache** — the cache of previously-computed key/value tensors that makes autoregressive generation avoid recomputing the whole sequence at every step (Project 13)
- **GGUF** — a file format for storing quantized model weights, used by `llama.cpp` (Project 27)
- **NTK / YaRN** — two related methods for extending a model's context length beyond what it was trained on (Project 16)
- **VLM** — Vision-Language Model (Project 29)
- **SSM** — State-Space Model, the mathematical family Mamba belongs to (Project 30)
