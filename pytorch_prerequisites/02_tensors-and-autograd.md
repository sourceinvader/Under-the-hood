# 2. Core: tensors, autograd, and `nn.Module` (required from Project 2 onward)

This is the load-bearing 20% of PyTorch that shows up in nearly every project. If you're new to PyTorch, this is what to learn first — a couple of hours with the official ["Deep Learning with PyTorch: A 60 Minute Blitz"](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) plus any "build a small classifier" tutorial covers essentially all of it. Math background for this section: [`01_math-foundations.md`](01_math-foundations.md) §1.1–1.3.

Every snippet below is deliberately tiny — a handful of scalars or a `(2, 3)`-shaped tensor at most. Run them as you read; the goal is to see the same idea from both directions, formula and output.

---

## Tensors

- Creating tensors: `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.randn`, `torch.arange`, `torch.full`, `torch.zeros_like` / `torch.ones_like`
- Dtypes: `torch.long`, `torch.float`, `torch.bool`, and why integer token IDs vs. float activations matter
- `.item()` to pull a Python scalar out of a 0-d tensor
- `.shape` and shape-unpacking idioms (`B, T, C = x.shape`)

A tensor's shape *is* the mathematical object it represents: a `(d,)` tensor is a vector, a `(d_out, d_in)` tensor is the matrix behind an `nn.Linear`, a `(B, T, d)` tensor is a batch of `B` sequences of `T` vectors each. Reading a project's tensor shapes as comments (`# (B, T, C)`) is reading the linear algebra directly — see §1.1 for what matmul does to these shapes.

```python
import torch

x = torch.zeros(2, 3)
print(x.shape, x.dtype)                        # torch.Size([2, 3]) torch.float32

token_ids = torch.tensor([5, 12, 3], dtype=torch.long)   # integer IDs: use torch.long
print(token_ids, token_ids.dtype)              # tensor([ 5, 12,  3]) torch.int64
```

## Autograd

- `requires_grad` / `requires_grad_()`, `.backward()`, `.grad`
- `torch.no_grad()` (context manager) and `@torch.no_grad()` (decorator) — both appear, used interchangeably, never contrasted in the book
- Why gradients accumulate by default and must be zeroed (`optimizer.zero_grad()` or `p.grad = None`) before each `.backward()`
- Why in-place ops on a `requires_grad=True` leaf tensor outside `no_grad()` are unsafe — the book exploits this correctly (e.g. freezing embedding rows in Project 2) but never explains the underlying rule

**What `.backward()` actually computes.** Setting `requires_grad=True` tells PyTorch to record every operation a tensor participates in, building a graph of *local* derivatives as the forward pass runs. Calling `.backward()` walks that graph in reverse, applying the chain rule at each node (§1.2, with a worked example and runnable code) to accumulate `∂loss/∂θ` into `θ.grad` for every leaf parameter `θ`. Project 1's `Value` class does this explicitly, node by node, in pure Python — `torch.Tensor.backward()` is the same algorithm, just compiled and running on whole tensors instead of individual scalars. If you've worked through Project 1's `_backward` closures and its topological-sort-then-reverse-traversal logic, you have already implemented the mechanism every later `.backward()` call in the book relies on.

**Why gradients accumulate, concretely.** Project 2's tied-weight pattern (`self.lm_head.weight = self.token_embedding.weight`, formalized in Project 5) is the clearest illustration: the same `Parameter` object is read in two different places in the forward pass, so by the multivariable chain rule (§1.2) its correct total gradient is the *sum* of the local gradients from both usage sites.

```python
import torch

a = torch.tensor(3.0, requires_grad=True)
b = a + a                  # a is used twice
b.backward()
print(a.grad)              # tensor(2.)  -- one contribution from each use of a

a.grad = None              # the equivalent of optimizer.zero_grad() for this one tensor
c = a * 5
c.backward()
print(a.grad)              # tensor(5.)  -- clean, single contribution

# now the same two backward() calls WITHOUT zeroing in between:
a2 = torch.tensor(3.0, requires_grad=True)
(a2 + a2).backward()
(a2 * 5).backward()
print(a2.grad)             # tensor(7.)  == 2.0 + 5.0 -- the two calls' grads summed together
```

`.grad`'s accumulate-by-default behavior isn't an implementation quirk — it's the only way `.backward()` can give a mathematically correct answer when the same tensor feeds into a computation more than once, which is why `optimizer.zero_grad()` at the top of every training loop is load-bearing rather than boilerplate: skip it, and the last line above is exactly the bug you'd introduce.

**Why in-place mutation is hazardous, concretely.** Some backward formulas need a forward-pass *value*, not just the operation — e.g. `d(tanh(x))/dx = 1 - tanh(x)²` needs the actual output of `tanh(x)`. If you overwrite that tensor in place before `.backward()` runs, the value autograd needs may already be gone, and PyTorch will either raise an error (if it detects the hazard) or — worse — silently compute a wrong gradient. `torch.no_grad()` avoids the question entirely by never building the graph, which is exactly why Project 2's `break_it.py` wraps its in-place embedding-row collapse in `with torch.no_grad(): ...`.

## Broadcasting and indexing

- Standard NumPy-style broadcasting rules — used constantly and never re-derived after Project 2
- Basic slicing (`x[:, -1, :]`) and fancy/advanced indexing (a tensor of integers as an index, e.g. `self.C[X]` in Project 2, `h[torch.arange(B), lengths - 1]` in Project 25)
- Boolean masks and `.masked_fill(mask, value)` — the `masked_fill(mask == 0, float("-inf"))`-then-`softmax` pattern for causal masking is used in at least 6 projects and never re-explained after its first appearance
- `.gather(dim, index)` for per-row lookups (picking out one log-probability per token) — a recurring, non-obvious idiom from Project 2 onward, especially dense in Projects 22–25

**The broadcasting rule, precisely.** Compare two shapes from the right (trailing dimension first). Two dimensions are compatible if they're equal, or if either one is `1` (it gets stretched to match). Missing leading dimensions are treated as `1`. So `(B, T, D) + (D,)` works (the `(D,)` vector is broadcast across every row of every batch element) but `(B, T, D) + (T,)` does not (a size-`T` vector can't align against a trailing size-`D` axis unless `T == D`).

**`.gather`, precisely.** `out[i][j] = input[i][index[i][j]]` (for `dim=1`; other dims generalize the pattern) — it picks one element per row using a *per-row* index, rather than a single index shared by every row (which is all plain slicing can do).

```python
import torch
import torch.nn.functional as F

mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)   # (2, 3) -- causal-style mask
scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])        # (2, 3)
masked = scores.masked_fill(~mask, float("-inf"))
print(masked)
# tensor([[1., 2., -inf], [4., -inf, -inf]])
print(F.softmax(masked, dim=-1))
# tensor([[0.2689, 0.7311, 0.0000], [1.0000, 0.0000, 0.0000]])  -- masked positions get exactly 0 probability

logp = torch.log(torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]]))
targets = torch.tensor([[1], [2]])          # "the log-prob I want differs per row"
print(logp.gather(1, targets).squeeze(-1))  # tensor([-0.3567, -0.9163])
print(logp[0, 1].item(), logp[1, 2].item()) # -0.3567 -0.9163 -- gather picked exactly these
```

This is exactly the tool you need for "the log-probability of *this specific* token, which differs for every position in the batch" — the recurring `log_softmax(...).gather(-1, targets.unsqueeze(-1))` idiom throughout Projects 22–25.

## Shape manipulation

- `.view()` vs `.reshape()` vs `.transpose()` vs `.permute()`, and critically: **why `.contiguous()` is required before `.view()` after a `.transpose()`** (transpose returns a strided view, not a copy; view demands contiguous memory). This exact pattern — `.transpose(1,2).contiguous().view(...)` — appears in Projects 4, 5, 7, 15, and more, and the book never once explains it in code or prose. If you don't already know this, it will look like superstition.
- `.unsqueeze()` / `.squeeze()` / `.expand()` / `.flatten()`
- The `expand()`-then-`reshape()` trick specifically (Project 15's `repeat_kv` for grouped-query attention) relies on knowing that `expand` is a zero-copy, stride-0 broadcast and `reshape` may force a copy afterward — a subtler variant of the point above

**Why `.contiguous()` is needed, concretely.** A tensor's data lives in one flat block of memory; its shape and *strides* (how many elements to skip to move one step along each dimension) describe how to read that flat block as a multi-dimensional array. `.transpose()` doesn't move any data — it just swaps two strides, so the tensor is still "the same data, read in a different order." `.view()` requires the requested new shape to be expressible as a single reinterpretation of the *existing* strides, which a transposed layout usually can't satisfy:

```python
import torch

t = torch.arange(12).reshape(3, 4)
tt = t.transpose(0, 1)
print(tt.is_contiguous())     # False -- transpose only swapped strides, didn't move data

try:
    tt.view(12)
except RuntimeError as e:
    print(e)   # "view size is not compatible with input tensor's size and stride ..."

print(tt.contiguous().view(12).shape)   # torch.Size([12]) -- fine, after a real copy
print(tt.reshape(12).shape)             # torch.Size([12]) -- reshape() does the copy for you when needed
```

`.expand()` is the more extreme version of the same idea: it sets a dimension's stride to `0` so the *same* underlying values are read repeatedly without copying anything:

```python
import torch

H_kv, T, d, n_rep = 2, 3, 4, 3          # 2 KV heads, repeated 3x each -> 6 query heads
x = torch.arange(H_kv * T * d).float().reshape(H_kv, T, d)

expanded = x[:, None, :, :].expand(H_kv, n_rep, T, d)
print(expanded.stride())                 # (12, 0, 4, 1) -- the 0 is the giveaway: no copy yet
reshaped = expanded.reshape(H_kv * n_rep, T, d)   # this line is where the copy happens
print(reshaped.shape)                    # torch.Size([6, 3, 4])
print(torch.equal(reshaped[0], reshaped[1]), torch.equal(reshaped[1], reshaped[2]))
# True True -- confirms the same original KV head got repeated 3 times, not 3 different heads
```

This is exactly how `repeat_kv` (Project 15) turns `n_kv_head` key/value heads into `n_head` heads for free — until the subsequent `.reshape()` forces a real copy because a `0`-stride dimension can't be flattened into its neighbor without one.

## `nn.Module` basics

(introduced wholesale in Project 5, assumed fluently thereafter)

- Subclassing `nn.Module`, calling `super().__init__()`, and how attribute assignment in `__init__` auto-registers submodules and parameters
- The `__init__` / `forward` contract; calling a module instance invokes `forward`
- `nn.Parameter` vs. a plain tensor vs. a **buffer** (`self.register_buffer(...)`) — buffers move with `.to(device)` and persist in `state_dict()` but don't get gradients and aren't updated by the optimizer (used for causal masks). This three-way distinction is used correctly throughout but is never spelled out anywhere in the repo.
- Common layers: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`, `nn.Sequential`, `nn.ModuleList`, `nn.GELU` / `nn.SiLU`
- `model.parameters()`, `p.numel()` for parameter counting — including the tied-weight double-counting trap (see the main [`README.md`](README.md#repo-specific-reading-notes))
- Weight tying by direct attribute aliasing (`self.lm_head.weight = self.token_embedding.weight`) — two attributes referencing the same `Parameter` object, so gradients from both usage sites accumulate onto one tensor (the concrete case for the "why gradients accumulate" point above)

```python
import torch
import torch.nn as nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(2))              # trainable
        self.register_buffer("mask", torch.tensor([1., 0.]))  # moves/saves with the model, never trained
        self.plain = torch.tensor([9., 9.])                 # NOT registered -- invisible to PyTorch

m = Tiny()
print([n for n, _ in m.named_parameters()])   # ['w']
print(list(m.state_dict().keys()))            # ['w', 'mask']  -- buffer is saved, plain tensor is not
```

And weight tying, with the gradient-accumulation point from the Autograd section made concrete on a real `nn.Module`:

```python
import torch
import torch.nn as nn

class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 4)
        self.head = nn.Linear(4, 5, bias=False)
        self.head.weight = self.emb.weight    # same Parameter object, not a copy

    def forward(self, idx):
        return self.head(self.emb(idx))       # weight used at BOTH the input and output side

m = Tied()
m(torch.tensor([0, 1])).sum().backward()
print(m.emb.weight.grad is m.head.weight.grad)   # True -- there's only one tensor, so only one .grad
```

## The functional API

- `import torch.nn.functional as F`: `F.softmax`, `F.cross_entropy` (raw logits in, integer class-index targets out — used as a black box), `F.log_softmax`, `F.logsigmoid`, `F.silu`

These are exactly the formulas from [`01_math-foundations.md`](01_math-foundations.md) §1.5, applied elementwise or row-wise to real tensors (runnable example there): `F.softmax(x, dim=-1)_i = exp(x_i) / Σ_j exp(x_j)`; `F.cross_entropy(logits, target)` internally computes `-F.log_softmax(logits, dim=-1)` and gathers out the `target` index; `F.logsigmoid(x) = -log(1+exp(-x))`, computed stably rather than as literal `log(sigmoid(x))`; `F.silu(x) = x * sigmoid(x)`.

## Optimizers and the training loop

- `torch.optim.AdamW` — construction, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` — appears in essentially every training loop from Project 5 onward (formula and runnable example: [`01_math-foundations.md`](01_math-foundations.md) §1.6)
- `model.eval()` vs `model.train()` and why it matters for dropout/normalization layers even though most runs in this repo use `dropout=0.0` (which silently papers over an actual inconsistency in `build.py` itself — see the main [`README.md`](README.md#repo-specific-reading-notes))

**What `AdamW` actually computes**, per parameter `θ`, per step `t`, given gradient `g_t`:

```
m_t = β1 * m_{t-1} + (1 - β1) * g_t          # running mean of the gradient (momentum)
v_t = β2 * v_{t-1} + (1 - β2) * g_t²          # running mean of the squared gradient (adaptive scale)
m̂_t = m_t / (1 - β1^t)                        # bias correction (m, v start at 0, so early steps are biased low)
v̂_t = v_t / (1 - β2^t)
θ_t = θ_{t-1} - lr * ( m̂_t / (√v̂_t + ε) + λ * θ_{t-1} )
```

The last line's `+ λ * θ_{t-1}` term is *decoupled weight decay* — the "W" in AdamW. It shrinks every parameter directly, rather than folding an L2 penalty into `g_t` before the momentum/variance running averages see it (which is what plain `Adam` with `weight_decay` does, and which the AdamW paper showed interacts badly with the adaptive per-parameter learning rate `1/√v̂_t`). This is also the exact reason Project 6 splits parameters into two optimizer groups (decayed vs. not) — bias and normalization parameters shouldn't be shrunk toward zero the way weight matrices should. Written out by hand for a single step, this matches `torch.optim.AdamW` exactly:

```python
import torch

theta = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
opt = torch.optim.AdamW([theta], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
theta.grad = torch.tensor([0.5, -0.2])
opt.step()
print(theta.data)     # tensor([0.8990, 1.0990])

# the same single step, computed by hand from the four equations above:
theta2, m0, v0 = torch.tensor([1.0, 1.0]), torch.zeros(2), torch.zeros(2)
beta1, beta2, lr, eps, wd = 0.9, 0.999, 0.1, 1e-8, 0.01
g = torch.tensor([0.5, -0.2])
m1, v1 = beta1 * m0 + (1 - beta1) * g, beta2 * v0 + (1 - beta2) * g**2
m_hat, v_hat = m1 / (1 - beta1**1), v1 / (1 - beta2**1)
theta2 = theta2 - lr * (m_hat / (v_hat.sqrt() + eps) + wd * theta2)
print(theta2)         # tensor([0.8990, 1.0990]) -- identical to opt.step() above
```

## Reproducibility

- `torch.manual_seed()` (global) used *alongside* an explicit `torch.Generator().manual_seed()` (local, passed as `generator=` to individual ops) — both appear together from Project 2 on, and the relationship between the two is never discussed

Both seed the same kind of pseudo-random number generator (PRNG); the difference is scope. `torch.manual_seed()` reseeds PyTorch's one global default generator, affecting every subsequent random call anywhere in the process unless told otherwise. A `torch.Generator().manual_seed()` instance is an independent PRNG state you pass explicitly (`generator=g`) to individual calls — useful when you want one specific random stream (say, sampling from a model) to be reproducible independent of how many *other* random calls happened elsewhere first.

```python
import torch

torch.manual_seed(0)
a = torch.randn(3)
torch.manual_seed(0)
b = torch.randn(3)
print(torch.equal(a, b))    # True -- same global seed, same values

g1 = torch.Generator().manual_seed(42)
g2 = torch.Generator().manual_seed(42)
c = torch.randn(3, generator=g1)
d = torch.randn(3, generator=g2)
print(torch.equal(c, d))    # True -- independent generator objects, same seed, same values regardless of global state
```

---

## Glossary

- **Leaf tensor** — a tensor you created directly (e.g. `torch.tensor(...)`, `nn.Parameter(...)`), as opposed to one produced by an operation on other tensors. Only leaf tensors accumulate `.grad`; intermediate results don't, by default.
- **Computation graph** — the record of operations autograd builds during the forward pass, used in reverse during `.backward()`.
- **In-place operation** — one that modifies a tensor's existing memory instead of returning a new tensor (PyTorch marks these with a trailing underscore, e.g. `.add_()` vs `.add()`).
- **Stride** — how many elements to skip in memory to move one step along a given dimension; what `.transpose()` changes without touching the underlying data.
- **Contiguous** — a tensor whose strides match a straightforward left-to-right reading of its shape; required by `.view()`.
- **Buffer** — a named tensor attached to an `nn.Module` (via `register_buffer`) that isn't a parameter: no gradient, no optimizer updates, but still moves with `.to(device)` and saves in `state_dict()`.
- **Weight tying** — making two parts of a model share the literal same weight tensor rather than two separately-learned copies.
- **Learning rate (`lr`)** — the step size an optimizer takes in the direction that reduces the loss.
- **Momentum** — carrying over a running average of past gradients so an optimizer's updates smooth out noise instead of reacting to every single gradient independently.
- **Weight decay** — a penalty that shrinks parameters toward zero each step, independent of the loss's own gradient; a way of discouraging unnecessarily large weights.

Next: [`03_systems-and-training.md`](03_systems-and-training.md) — module introspection, hooks, mixed precision, and distributed training (Tier 2, roughly Projects 5–17).
