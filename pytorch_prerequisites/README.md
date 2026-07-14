# PyTorch Prerequisites for *Under the Hood*

This document maps out exactly how much PyTorch you need to know to work through all 35 projects in this repository, and where each piece of knowledge first becomes necessary. It was produced by reading every project's `README.md`, `build.py`, and `step_*.py` files and cataloging the concrete PyTorch APIs used versus the concepts each project assumes you already have.

The book's own stated bar (`README.md`) is: *"You should be comfortable with Python and have seen a tensor before. Everything else is built up from scratch."* That's true in spirit, but "everything else" undersells it a little in the back half of the book — projects 9–35 lean on a fair amount of PyTorch idiom (hooks, `named_modules()`, distributed collectives, HuggingFace-style model conventions) that the book doesn't stop to teach because it isn't really about PyTorch at that point — it's about systems, algorithms, and math, and PyTorch is just the notation. This document is the missing "notation primer."

---

## TL;DR

- **To start Project 1:** no PyTorch at all. It's pure Python.
- **To start Project 2** (first project that imports `torch`): you need [Tier 1](#tier-1--core-required-from-project-2-onward) below — tensors, autograd basics, broadcasting, indexing, and eventually `nn.Module`.
- **To comfortably reach Project 17** (end of the systems/inference arc): add [Tier 2](#tier-2--intermediate-required-by-roughly-projects-5-17) — module introspection, hooks, mixed precision, device/CUDA management, and `torch.distributed`/FSDP.
- **To get through Projects 18–35** without constantly stopping to look things up: add [Tier 3](#tier-3--advanced-needed-for-specific-later-projects-18-35) — the pattern here is less "new PyTorch API" and more "the code becomes thinner and more like transcribed pseudocode," so the prerequisite shifts from *tensor mechanics* to *reading partially-specified code and filling in the undefined pieces yourself*.
- If you already know research-level PyTorch (you've trained a transformer from scratch before, you know what `requires_grad` and `register_buffer` do, you've used `torch.distributed` at least once), you can start on Project 1 today and treat this document as a reference rather than a study plan.

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

If you already know some of these, great — the book will feel like confirmation rather than discovery in those chapters. If you don't, that's the point: you're not expected to.

---

## Tier 1 — Core (required from Project 2 onward)

This is the load-bearing 20% of PyTorch that shows up in nearly every project. If you're new to PyTorch, this is what to learn first — a couple of hours with the official ["Deep Learning with PyTorch: A 60 Minute Blitz"](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) plus any "build a small classifier" tutorial covers essentially all of it.

**Tensors**
- Creating tensors: `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.randn`, `torch.arange`, `torch.full`, `torch.zeros_like` / `torch.ones_like`
- Dtypes: `torch.long`, `torch.float`, `torch.bool`, and why integer token IDs vs. float activations matter
- `.item()` to pull a Python scalar out of a 0-d tensor
- `.shape` and shape-unpacking idioms (`B, T, C = x.shape`)

**Autograd**
- `requires_grad` / `requires_grad_()`, `.backward()`, `.grad`
- `torch.no_grad()` (context manager) and `@torch.no_grad()` (decorator) — both appear, used interchangeably, never contrasted in the book
- Why gradients accumulate by default and must be zeroed (`optimizer.zero_grad()` or `p.grad = None`) before each `.backward()`
- Why in-place ops on a `requires_grad=True` leaf tensor outside `no_grad()` are unsafe — the book exploits this correctly (e.g. freezing embedding rows in Project 2) but never explains the underlying rule

**Broadcasting and indexing**
- Standard NumPy-style broadcasting rules — used constantly and never re-derived after Project 2
- Basic slicing (`x[:, -1, :]`) and fancy/advanced indexing (a tensor of integers as an index, e.g. `self.C[X]` in Project 2, `h[torch.arange(B), lengths - 1]` in Project 25)
- Boolean masks and `.masked_fill(mask, value)` — the `masked_fill(mask == 0, float("-inf"))`-then-`softmax` pattern for causal masking is used in at least 6 projects and never re-explained after its first appearance
- `.gather(dim, index)` for per-row lookups (picking out one log-probability per token) — a recurring, non-obvious idiom from Project 2 onward, especially dense in Projects 22–25

**Shape manipulation**
- `.view()` vs `.reshape()` vs `.transpose()` vs `.permute()`, and critically: **why `.contiguous()` is required before `.view()` after a `.transpose()`** (transpose returns a strided view, not a copy; view demands contiguous memory). This exact pattern — `.transpose(1,2).contiguous().view(...)` — appears in Projects 4, 5, 7, 15, and more, and the book never once explains it in code or prose. If you don't already know this, it will look like superstition.
- `.unsqueeze()` / `.squeeze()` / `.expand()` / `.flatten()`
- The `expand()`-then-`reshape()` trick specifically (Project 15's `repeat_kv` for grouped-query attention) relies on knowing that `expand` is a zero-copy, stride-0 broadcast and `reshape` may force a copy afterward — a subtler variant of the point above

**`nn.Module` basics** (introduced wholesale in Project 5, assumed fluently thereafter)
- Subclassing `nn.Module`, calling `super().__init__()`, and how attribute assignment in `__init__` auto-registers submodules and parameters
- The `__init__` / `forward` contract; calling a module instance invokes `forward`
- `nn.Parameter` vs. a plain tensor vs. a **buffer** (`self.register_buffer(...)`) — buffers move with `.to(device)` and persist in `state_dict()` but don't get gradients and aren't updated by the optimizer (used for causal masks). This three-way distinction is used correctly throughout but is never spelled out anywhere in the repo.
- Common layers: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`, `nn.Sequential`, `nn.ModuleList`, `nn.GELU` / `nn.SiLU`
- `model.parameters()`, `p.numel()` for parameter counting — including the tied-weight double-counting trap (see [Reading notes](#repo-specific-reading-notes) below)
- Weight tying by direct attribute aliasing (`self.lm_head.weight = self.token_embedding.weight`) — two attributes referencing the same `Parameter` object, so gradients from both usage sites accumulate onto one tensor

**The functional API**
- `import torch.nn.functional as F`: `F.softmax`, `F.cross_entropy` (raw logits in, integer class-index targets out — used as a black box), `F.log_softmax`, `F.logsigmoid`, `F.silu`

**Optimizers and the training loop**
- `torch.optim.AdamW` — construction, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` — appears in essentially every training loop from Project 5 onward
- `model.eval()` vs `model.train()` and why it matters for dropout/normalization layers even though most runs in this repo use `dropout=0.0` (which silently papers over an actual inconsistency in `build.py` itself — see below)

**Reproducibility**
- `torch.manual_seed()` (global) used *alongside* an explicit `torch.Generator().manual_seed()` (local, passed as `generator=` to individual ops) — both appear together from Project 2 on, and the relationship between the two is never discussed

---

## Tier 2 — Intermediate (required by roughly Projects 5–17)

**Module introspection**
- `model.named_modules()`, `model.modules()`, `model.named_parameters()`, and the non-obvious `module.parameters(recurse=False)` (a module's *own* parameters, not its children's) — used to selectively apply weight decay, custom init, or per-layer instrumentation by `isinstance` type-checking (`isinstance(m, nn.Linear)`, `isinstance(m, nn.LayerNorm)`, etc.)
- `id(param)`-based deduplication — required wherever you walk the module tree and a weight-tied parameter would otherwise get counted or classified twice

**Checkpointing**
- `model.state_dict()` / `load_state_dict()`, `torch.save` / `torch.load`

**Hooks**
- `module.register_forward_hook(fn)` with signature `fn(module, input, output)` — used for activation-statistics logging (Project 11); `.detach()` before computing stats inside a hook, so logging doesn't extend the autograd graph

**Mixed precision**
- `torch.cuda.amp.GradScaler()` and the `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()` pattern, plus reading `scaler.get_scale()` to detect a skipped step (overflow)
- `dtype=torch.float16` / `torch.bfloat16` passed explicitly to tensor constructors
- Worth knowing going in: real mixed-precision training normally pairs `GradScaler` with `torch.autocast(...)` wrapping the forward pass — **this repo never actually shows `autocast`**, only the `GradScaler` overflow-bookkeeping half. Don't expect to find a canonical autocast example here; bring that knowledge with you if you want the full picture.

**Device and CUDA management**
- `device=` kwargs on tensor constructors, `.to(device)`, `torch.cuda.is_available()`
- `torch.cuda.synchronize()` and *why* it's needed before timing anything (CUDA calls are asynchronous by default)
- `torch.cuda.max_memory_allocated()`, `reset_peak_memory_stats()`, `empty_cache()`, and catching `torch.cuda.OutOfMemoryError` — shown in Project 8's step files (not its `build.py`) for measuring FlashAttention's real memory advantage

**Batched attention tensor choreography**
- The `[B, H, T, d_head]` shape convention and `transpose(1, 2)` to split/merge the head dimension — this is assumed fluent by Project 5 and never re-taught
- Sampling and lookup ops: `torch.multinomial`, `torch.argmax`, `torch.topk`, `.gather()` for per-row probability lookups — dense in Projects 14 (speculative decoding) and 22/24/25 (evaluation, DPO, reasoning)

**Distributed training** (concentrated almost entirely in Project 12, which is the single most PyTorch-systems-heavy project in the book)
- `torch.distributed`: `init_process_group(backend=...)`, `destroy_process_group()`, `all_reduce`, `all_gather`, `reduce_scatter_tensor`, `ReduceOp.SUM`
- `torch.multiprocessing.spawn(fn, args=..., nprocs=...)` and its rank-injection convention (each spawned process calls `fn(rank, *args)`)
- `torch.distributed.fsdp.FullyShardedDataParallel` (FSDP) and `torch.distributed.fsdp.wrap.transformer_auto_wrap_policy` — wrapping each transformer block as its own FSDP unit rather than the whole model at once, and why that controls the granularity of the all-gather/reduce-scatter traffic
- Process groups, rank, world size, and backend choice (`"gloo"` for the CPU-friendly single-box proxy in this repo vs. `"nccl"` for real multi-GPU, which is never shown)

---

## Tier 3 — Advanced (needed for specific later projects, 18–35)

By this point in the book, the pattern changes: individual chapters introduce far *less* new PyTorch API surface, but the code itself gets thinner and more elliptical — several `build.py` files are closer to transcribed pseudocode (undefined helper functions like `policy.generate()`, `reward_model.score()`, `kl_to_reference()` in Project 23) than to runnable programs. The prerequisite shifts from "do you know this tensor op" to "can you read a sketch of an algorithm and supply the missing plumbing yourself, the way you'd read a paper's pseudocode." Knowing Tiers 1–2 solidly is what makes that possible.

- **MoE-style routing** (Project 18): `torch.topk(x, k, dim=-1)` for expert selection, `Tensor.scatter_()` for building a sparse combination mask, `torch.bincount` for utilization tracking. Note the book's reference implementation runs *every* expert on *every* token and masks afterward — the opposite of a real production MoE kernel's token-permutation approach — worth knowing that distinction going in so you don't mistake it for the efficient version.
- **Low-rank adapters / LoRA** (Project 21): wrapping an existing `nn.Module` inside a new one, `requires_grad_(False)` to freeze the base weight while still differentiating *through* it for upstream gradients, `nn.init.normal_` / `nn.init.zeros_` for the asymmetric adapter initialization that makes the adapter a no-op at step zero.
- **Preference-optimization losses — DPO/KTO/ORPO/SimPO** (Project 24): `F.logsigmoid` for numerically-stable log-sigmoid; the recurring "shift-by-one, `log_softmax`, `gather`, mask, sum" idiom for computing a sequence's log-probability under a model; running a frozen reference model (`torch.no_grad()`) alongside a trainable policy in the same step. One genuine landmine: ORPO's `torch.log(1 - torch.exp(logp))` is numerically fragile as `logp → 0` (confident predictions drive the argument toward 0, and the log toward `-inf`) — it's transcribed directly from the paper's formula without a stabilized rewrite, so don't assume it's a safe pattern to reuse elsewhere.
- **Policy-gradient / RLHF basics** (Project 23): `.detach()` to stop gradient flow into a baseline/advantage term while still letting it flow through the log-probability term — the mechanics of a REINFORCE-style surrogate loss, which looks nothing like the cross-entropy/MSE losses used everywhere earlier in the book. This project's code is the thinnest/most pseudocode-like in the repo, so treat the README's prose as the primary source and the code as a sketch.
- **Quantization** (Project 27): real dtype casts (`.to(torch.int8)`), `torch.round` / `torch.clamp` for the quantize/dequantize round trip, reading `.weight.data` directly (bypassing the `Parameter` wrapper, safe here specifically because nothing is being trained), and `named_modules()` traversal to quantize every `nn.Linear` generically. Worth knowing: this project always dequantizes back to float before computing, so it measures rounding error but never actually exercises a real low-bit matmul kernel.
- **Vision layers** (Project 29): `nn.Conv2d` used with `stride == kernel_size` as a non-overlapping patch-embedding trick, then `.flatten(2).transpose(1, 2)` to turn the conv output into a token sequence you can concatenate with text embeddings. Also the `ignore_index=-100` convention for `F.cross_entropy` to exclude image-token positions from the language-modeling loss — the one place in the book this label-masking convention is used, introduced with no walkthrough.
- **Custom recurrent / state-space blocks** (Project 30): heavy reliance on broadcasting across 4-D tensors, `torch.nn.functional.softplus` to keep a parameter positive, log-space parameterization (`torch.log`/`torch.exp`) for numerical stability, and — distinctively — a **manual sequential Python `for` loop over time steps** (not a vectorized scan), appending to a list and `torch.stack`-ing at the end. This is the only place in the book where the "recurrence" isn't fully parallelized, and it's worth recognizing that as a deliberate pedagogical simplification, not how production Mamba/RWKV kernels actually work. **Version note:** this project uses the built-in `nn.RMSNorm`, which requires **PyTorch ≥ 2.4**; the repo's own `requirements.txt` pins `torch>=2.2,<3.0`, so a minimal install following the setup docs literally can hit an `AttributeError` here.
- **Contrastive / retrieval losses** (Project 28): the InfoNCE pattern — an all-pairs similarity matrix (`query_embs @ passage_embs.T`) scored against `F.cross_entropy(sim, torch.arange(batch_size))`, i.e. reusing cross-entropy as a retrieval loss by treating "the matching pair is on the diagonal" as the label. Recognizing this pattern is the whole prerequisite; the rest of that project is FAISS and NumPy, not PyTorch.
- **Module-compliance / introspection patterns** (Project 33): not the generic `named_parameters()`/`state_dict()` APIs you'd expect, but cruder, more direct techniques — `type(module).__name__` string comparisons and hardcoded dotted-attribute access (`module.attn.q_proj.weight.shape`) — plus runtime behavioral probes (`torch.isnan(y).any()`, checking the RMS of an output tensor falls in a healthy range) as a check that static structure alone can't catch a silent `LayerNorm`-for-`RMSNorm` swap.

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

- **Never used PyTorch:** work through the official 60-minute blitz plus one "train a small classifier" tutorial before starting Project 2. That covers essentially all of Tier 1. Come back to Tier 2 organically when you reach Projects 11–12.
- **Comfortable with basic PyTorch (tensors, autograd, a simple `nn.Module`), but never touched distributed training, hooks, or quantization:** you can start immediately. Skim Tier 2 before Project 11 and Tier 3's quantization/state-space notes before Projects 27 and 30 so the version trap and the "always dequantizes to float" caveat don't surprise you mid-chapter.
- **Already comfortable at a research-engineering level:** start on Project 1 today and use this document as a lookup table (via the per-project table above) rather than a study plan — the "Repo-specific reading notes" section is probably the most useful part for you, since it flags the places where the repo's code diverges from what the book's prose promises.
