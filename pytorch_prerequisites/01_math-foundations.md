# 1. Math foundations

This is the math underneath the PyTorch calls — not a full course, just the specific tools that keep reappearing across all 35 projects. If you already know this, skip straight to [`02_tensors-and-autograd.md`](02_tensors-and-autograd.md). If your math is rusty, read this first; every later file links back to the relevant section here instead of re-deriving it.

---

## 1.1 Vectors, matrices, and matrix multiplication

A tensor is just an n-dimensional array of numbers. The operation that dominates every model in this book is matrix multiplication: for `A` of shape `(m, n)` and `B` of shape `(n, p)`, `A @ B` has shape `(m, p)`, and entry `(i, j)` is the dot product of row `i` of `A` with column `j` of `B`:

```
(A @ B)[i, j] = Σ_k A[i, k] * B[k, j]
```

Every `nn.Linear(in_features, out_features)` is exactly this: a weight matrix `W` of shape `(out_features, in_features)` and `y = x @ W.T + b`. When a shape error crashes a forward pass, it's almost always this rule being violated — the inner dimensions (`n` above) don't match.

## 1.2 Derivatives, gradients, and the chain rule

A derivative `dy/dx` measures how much `y` changes for a small change in `x`. A gradient is the same idea generalized to a function of many variables — `∇f` is the vector of partial derivatives, one per input.

The rule that makes backpropagation possible is the **chain rule**: if `L` depends on `x` only through an intermediate value `z`, then

```
dL/dx = (dL/dz) * (dz/dx)
```

This is the entire mechanism Project 1's `Value` class automates, and it's what every `.backward()` call in every later project is doing under the hood — just with tensors instead of scalars, and graphs with thousands of nodes instead of two.

**Worked example** (mirrors Project 1's very first lines): let `z = x * y` and `loss = z ** 2`.

```
dz/dx = y                dz/dy = x
d(loss)/dz = 2z

d(loss)/dx = d(loss)/dz * dz/dx = 2z * y
d(loss)/dy = d(loss)/dz * dz/dx = 2z * x
```

`loss.backward()` computes exactly this, right to left, for every parameter in the graph, no matter how deep. This is why `requires_grad=True` and `.grad` exist: PyTorch records each operation's *local* derivative (`dz/dx = y`, `d(loss)/dz = 2z`, etc.) during the forward pass, then multiplies them together in reverse order during `.backward()`.

**Why gradients accumulate.** If a variable is used in more than one place in the graph, its total derivative is the *sum* over every path (the multivariable chain rule). That's why every `_backward` closure in Project 1 uses `+=`, not `=`, and why real tensors' `.grad` also accumulates across `.backward()` calls — it's not a PyTorch quirk, it's what the math requires. It's also exactly why you must call `optimizer.zero_grad()` (or `p.grad = None`) before each new `.backward()`: otherwise you're adding this step's gradient onto last step's leftover sum.

**Why in-place ops on tracked tensors are risky.** Some local derivatives need the *forward* value to compute the backward pass — e.g. `d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))` needs `sigmoid(x)` itself, not just `x`. If you overwrite a tensor in place after using it in the forward pass, autograd may no longer have the value it needs when `.backward()` runs later. `torch.no_grad()` sidesteps the whole issue by not building a graph in the first place.

## 1.3 Exponentials, logarithms, and the log-sum-exp trick

`exp` and `log` show up constantly (softmax, cross-entropy, log-probabilities) because they turn multiplication into addition: `log(a * b) = log(a) + log(b)`. Two consequences matter throughout this book:

- **Overflow.** `exp(x)` for even moderately large `x` (say, `x > 100`) overflows floating point. Anywhere you see a running `max` subtracted before an `exp` — Project 8's online softmax, `F.log_softmax`'s internal implementation — that's this problem being handled: `log Σ exp(x_i) = m + log Σ exp(x_i - m)` for `m = max(x_i)`, which is mathematically identical but numerically safe.
- **Underflow.** `log(p)` for `p` close to 0 goes to `-∞`. Project 24's ORPO loss (`torch.log(1 - torch.exp(logp))`) hits exactly this failure mode when `logp → 0` (a confident prediction) — see [`04_advanced-topics.md`](04_advanced-topics.md) for the concrete case.

## 1.4 Probability, likelihood, and negative log-likelihood

A language model doesn't output a single answer — it outputs a probability distribution over the vocabulary, `p_θ(next token | context)`. Training means adjusting `θ` so the model assigns high probability to the tokens that actually appeared in the training data. "High probability" is inconvenient to optimize directly (products of many small numbers underflow), so instead you maximize the **log**-probability, which turns the product of per-token probabilities into a sum:

```
log p_θ(y_1, ..., y_T | x) = Σ_t log p_θ(y_t | x, y_<t)
```

Flip the sign and you get the **negative log-likelihood (NLL)** — a quantity you minimize instead of maximizing likelihood, purely so "loss goes down" reads the natural way. Every `F.cross_entropy` call in this book is computing an NLL.

## 1.5 Softmax and cross-entropy

Softmax turns a vector of arbitrary real numbers ("logits") into a probability distribution:

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

Cross-entropy between the true label and the model's predicted distribution, when the true label is a single correct class `c`, reduces to just the negative log-probability the model assigned to `c`:

```
cross_entropy(logits, c) = -log( softmax(logits)_c ) = -log_softmax(logits)_c
```

That's why `F.cross_entropy` takes raw logits and an integer class index rather than a probability distribution — internally it computes `log_softmax` (using the log-sum-exp trick above for stability) and then just indexes out position `c`. This single formula is the loss function for every next-token-prediction training loop in the book, from Project 2 through Project 9, and reappears as the retrieval loss in Project 28 (see [`04_advanced-topics.md`](04_advanced-topics.md)).

`F.logsigmoid(x) = -log(1 + exp(-x))`, computed in a numerically stable way (not literally `log(sigmoid(x))`, which would underflow for very negative `x`). It shows up throughout Project 24's preference-optimization losses because `sigmoid` is the standard way to turn a real-valued "score difference" into something loss-shaped.

## 1.6 Norms and gradient clipping

The L2 norm of a vector (or, flattened, a tensor) is `||v||₂ = sqrt(Σ_i v_i²)` — its Euclidean length. `torch.nn.utils.clip_grad_norm_` computes the L2 norm of *all* parameter gradients concatenated together, and if it exceeds `max_norm`, rescales every gradient by the same factor so the total norm becomes exactly `max_norm`:

```
total_norm = sqrt(Σ_p ||grad_p||₂²)          # one number, across every parameter p
if total_norm > max_norm:
    for p: grad_p *= max_norm / total_norm
```

This is why one call clips *all* gradients together rather than each tensor independently — a single very large gradient in one layer would otherwise dominate and get clipped alone, while the relative direction of the whole gradient vector is what clipping is trying to preserve.

---

Next: [`02_tensors-and-autograd.md`](02_tensors-and-autograd.md) — the Tier 1 PyTorch surface, with these formulas applied to actual tensor code.
