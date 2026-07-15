# 2. Core: tensors, autograd, and `nn.Module` (required from Project 2 onward)

This is the load-bearing 20% of PyTorch that shows up in nearly every project. If you're new to PyTorch, this is what to learn first — a couple of hours with the official ["Deep Learning with PyTorch: A 60 Minute Blitz"](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) plus any "build a small classifier" tutorial covers essentially all of it. Math background for this section: [`01_math-foundations.md`](01_math-foundations.md) §1.1–1.3.

---

## Tensors

- Creating tensors: `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.randn`, `torch.arange`, `torch.full`, `torch.zeros_like` / `torch.ones_like`
- Dtypes: `torch.long`, `torch.float`, `torch.bool`, and why integer token IDs vs. float activations matter
- `.item()` to pull a Python scalar out of a 0-d tensor
- `.shape` and shape-unpacking idioms (`B, T, C = x.shape`)

A tensor's shape *is* the mathematical object it represents: a `(d,)` tensor is a vector, a `(d_out, d_in)` tensor is the matrix behind an `nn.Linear`, a `(B, T, d)` tensor is a batch of `B` sequences of `T` vectors each. Reading a project's tensor shapes as comments (`# (B, T, C)`) is reading the linear algebra directly — see §1.1 for what matmul does to these shapes.

## Autograd

- `requires_grad` / `requires_grad_()`, `.backward()`, `.grad`
- `torch.no_grad()` (context manager) and `@torch.no_grad()` (decorator) — both appear, used interchangeably, never contrasted in the book
- Why gradients accumulate by default and must be zeroed (`optimizer.zero_grad()` or `p.grad = None`) before each `.backward()`
- Why in-place ops on a `requires_grad=True` leaf tensor outside `no_grad()` are unsafe — the book exploits this correctly (e.g. freezing embedding rows in Project 2) but never explains the underlying rule

**What `.backward()` actually computes.** Setting `requires_grad=True` tells PyTorch to record every operation a tensor participates in, building a graph of *local* derivatives as the forward pass runs. Calling `.backward()` walks that graph in reverse, applying the chain rule at each node (§1.2) to accumulate `∂loss/∂θ` into `θ.grad` for every leaf parameter `θ`. Project 1's `Value` class does this explicitly, node by node, in pure Python — `torch.Tensor.backward()` is the same algorithm, just compiled and running on whole tensors instead of individual scalars. If you've worked through Project 1's `_backward` closures and its topological-sort-then-reverse-traversal logic, you have already implemented the mechanism every later `.backward()` call in the book relies on.

**Why gradients accumulate, concretely.** Project 2's tied-weight pattern (`self.lm_head.weight = self.token_embedding.weight`, formalized in Project 5) is the clearest illustration: the same `Parameter` object is read in two different places in the forward pass, so by the multivariable chain rule (§1.2) its correct total gradient is the *sum* of the local gradients from both usage sites. `.grad`'s accumulate-by-default behavior isn't an implementation quirk — it's the only way `.backward()` can give a mathematically correct answer when the same tensor feeds into a computation more than once, which is why `optimizer.zero_grad()` at the top of every training loop is load-bearing rather than boilerplate.

**Why in-place mutation is hazardous, concretely.** Some backward formulas need a forward-pass *value*, not just the operation — e.g. `d(tanh(x))/dx = 1 - tanh(x)²` needs the actual output of `tanh(x)`. If you overwrite that tensor in place before `.backward()` runs, the value autograd needs may already be gone, and PyTorch will either raise an error (if it detects the hazard) or — worse — silently compute a wrong gradient. `torch.no_grad()` avoids the question entirely by never building the graph, which is exactly why Project 2's `break_it.py` wraps its in-place embedding-row collapse in `with torch.no_grad(): ...`.

## Broadcasting and indexing

- Standard NumPy-style broadcasting rules — used constantly and never re-derived after Project 2
- Basic slicing (`x[:, -1, :]`) and fancy/advanced indexing (a tensor of integers as an index, e.g. `self.C[X]` in Project 2, `h[torch.arange(B), lengths - 1]` in Project 25)
- Boolean masks and `.masked_fill(mask, value)` — the `masked_fill(mask == 0, float("-inf"))`-then-`softmax` pattern for causal masking is used in at least 6 projects and never re-explained after its first appearance
- `.gather(dim, index)` for per-row lookups (picking out one log-probability per token) — a recurring, non-obvious idiom from Project 2 onward, especially dense in Projects 22–25

**The broadcasting rule, precisely.** Compare two shapes from the right (trailing dimension first). Two dimensions are compatible if they're equal, or if either one is `1` (it gets stretched to match). Missing leading dimensions are treated as `1`. So `(B, T, D) + (D,)` works (the `(D,)` vector is broadcast across every row of every batch element) but `(B, T, D) + (T,)` does not (a size-`T` vector can't align against a trailing size-`D` axis unless `T == D`). This single rule is what makes `masked_fill(mask == 0, ...)` work when `mask` is `(T, T)` and `scores` is `(B, H, T, T)` — the mask is broadcast identically across every batch element and every head.

**`.gather`, precisely.** `out[i][j] = input[i][index[i][j]]` (for `dim=1`; other dims generalize the pattern) — it picks one element per row using a *per-row* index, rather than a single index shared by every row (which is all plain slicing can do). This is exactly the tool you need for "the log-probability of *this specific* token, which differs for every position in the batch" — the recurring `log_softmax(...).gather(-1, targets.unsqueeze(-1))` idiom throughout Projects 22–25.

## Shape manipulation

- `.view()` vs `.reshape()` vs `.transpose()` vs `.permute()`, and critically: **why `.contiguous()` is required before `.view()` after a `.transpose()`** (transpose returns a strided view, not a copy; view demands contiguous memory). This exact pattern — `.transpose(1,2).contiguous().view(...)` — appears in Projects 4, 5, 7, 15, and more, and the book never once explains it in code or prose. If you don't already know this, it will look like superstition.
- `.unsqueeze()` / `.squeeze()` / `.expand()` / `.flatten()`
- The `expand()`-then-`reshape()` trick specifically (Project 15's `repeat_kv` for grouped-query attention) relies on knowing that `expand` is a zero-copy, stride-0 broadcast and `reshape` may force a copy afterward — a subtler variant of the point above

**Why `.contiguous()` is needed, concretely.** A tensor's data lives in one flat block of memory; its shape and *strides* (how many elements to skip to move one step along each dimension) describe how to read that flat block as a multi-dimensional array. `.transpose()` doesn't move any data — it just swaps two strides, so the tensor is still "the same data, read in a different order." `.view()` requires the requested new shape to be expressible as a single reinterpretation of the *existing* strides, which a transposed layout usually can't satisfy — hence the `RuntimeError` `.contiguous()` prevents by physically copying the data into the new order first. `.expand()` is the more extreme version of the same idea: it sets a dimension's stride to `0` so the *same* underlying values are read repeatedly without copying anything, which is exactly how `repeat_kv` (Project 15) turns `n_kv_head` key/value heads into `n_head` heads for free — until the subsequent `.reshape()` forces a real copy because a `0`-stride dimension can't be flattened into its neighbor without one.

## `nn.Module` basics

(introduced wholesale in Project 5, assumed fluently thereafter)

- Subclassing `nn.Module`, calling `super().__init__()`, and how attribute assignment in `__init__` auto-registers submodules and parameters
- The `__init__` / `forward` contract; calling a module instance invokes `forward`
- `nn.Parameter` vs. a plain tensor vs. a **buffer** (`self.register_buffer(...)`) — buffers move with `.to(device)` and persist in `state_dict()` but don't get gradients and aren't updated by the optimizer (used for causal masks). This three-way distinction is used correctly throughout but is never spelled out anywhere in the repo.
- Common layers: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`, `nn.Sequential`, `nn.ModuleList`, `nn.GELU` / `nn.SiLU`
- `model.parameters()`, `p.numel()` for parameter counting — including the tied-weight double-counting trap (see the main [`README.md`](README.md#repo-specific-reading-notes))
- Weight tying by direct attribute aliasing (`self.lm_head.weight = self.token_embedding.weight`) — two attributes referencing the same `Parameter` object, so gradients from both usage sites accumulate onto one tensor (the concrete case for the "why gradients accumulate" point above)

## The functional API

- `import torch.nn.functional as F`: `F.softmax`, `F.cross_entropy` (raw logits in, integer class-index targets out — used as a black box), `F.log_softmax`, `F.logsigmoid`, `F.silu`

These are exactly the formulas from [`01_math-foundations.md`](01_math-foundations.md) §1.5, applied elementwise or row-wise to real tensors: `F.softmax(x, dim=-1)_i = exp(x_i) / Σ_j exp(x_j)`; `F.cross_entropy(logits, target)` internally computes `-F.log_softmax(logits, dim=-1)` and gathers out the `target` index; `F.logsigmoid(x) = -log(1+exp(-x))`, computed stably rather than as literal `log(sigmoid(x))`; `F.silu(x) = x * sigmoid(x)`.

## Optimizers and the training loop

- `torch.optim.AdamW` — construction, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` — appears in essentially every training loop from Project 5 onward (formula: [`01_math-foundations.md`](01_math-foundations.md) §1.6)
- `model.eval()` vs `model.train()` and why it matters for dropout/normalization layers even though most runs in this repo use `dropout=0.0` (which silently papers over an actual inconsistency in `build.py` itself — see the main [`README.md`](README.md#repo-specific-reading-notes))

**What `AdamW` actually computes**, per parameter `θ`, per step `t`, given gradient `g_t`:

```
m_t = β1 * m_{t-1} + (1 - β1) * g_t          # running mean of the gradient (momentum)
v_t = β2 * v_{t-1} + (1 - β2) * g_t²          # running mean of the squared gradient (adaptive scale)
m̂_t = m_t / (1 - β1^t)                        # bias correction (m, v start at 0, so early steps are biased low)
v̂_t = v_t / (1 - β2^t)
θ_t = θ_{t-1} - lr * ( m̂_t / (√v̂_t + ε) + λ * θ_{t-1} )
```

The last line's `+ λ * θ_{t-1}` term is *decoupled weight decay* — the "W" in AdamW. It shrinks every parameter directly, rather than folding an L2 penalty into `g_t` before the momentum/variance running averages see it (which is what plain `Adam` with `weight_decay` does, and which the AdamW paper showed interacts badly with the adaptive per-parameter learning rate `1/√v̂_t`). This is also the exact reason Project 6 splits parameters into two optimizer groups (decayed vs. not) — bias and normalization parameters shouldn't be shrunk toward zero the way weight matrices should.

## Reproducibility

- `torch.manual_seed()` (global) used *alongside* an explicit `torch.Generator().manual_seed()` (local, passed as `generator=` to individual ops) — both appear together from Project 2 on, and the relationship between the two is never discussed

Both seed the same kind of pseudo-random number generator (PRNG); the difference is scope. `torch.manual_seed()` reseeds PyTorch's one global default generator, affecting every subsequent random call anywhere in the process unless told otherwise. A `torch.Generator().manual_seed()` instance is an independent PRNG state you pass explicitly (`generator=g`) to individual calls (`torch.randn(..., generator=g)`, `torch.multinomial(..., generator=g)`) — useful when you want one specific random stream (say, sampling from a model) to be reproducible independent of how many *other* random calls happened elsewhere first.

---

Next: [`03_systems-and-training.md`](03_systems-and-training.md) — module introspection, hooks, mixed precision, and distributed training (Tier 2, roughly Projects 5–17).
