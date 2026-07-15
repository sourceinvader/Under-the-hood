# 4. Advanced: MoE, LoRA, preference optimization, RLHF, quantization, and beyond-transformer architectures (Projects 18–35)

By this point in the book, the pattern changes: individual chapters introduce far *less* new PyTorch API surface, but the code itself gets thinner and more elliptical — several `build.py` files are closer to transcribed pseudocode (undefined helper functions like `policy.generate()`, `reward_model.score()`, `kl_to_reference()` in Project 23) than to runnable programs. The prerequisite shifts from "do you know this tensor op" to "can you read a sketch of an algorithm and supply the missing plumbing yourself, the way you'd read a paper's pseudocode." Knowing [`02_tensors-and-autograd.md`](02_tensors-and-autograd.md) and [`03_systems-and-training.md`](03_systems-and-training.md) solidly is what makes that possible. Every formula below is transcribed directly from this repo's actual `build.py`, not from the general literature, so it matches what you'll find in the code exactly.

---

## Mixture-of-Experts routing (Project 18)

`torch.topk`, `Tensor.scatter_()`, `torch.bincount`. The router is one `nn.Linear(d_model, num_experts)`, and the whole layer is:

```
router_probs = softmax(router(x), dim=-1)                  # (B, T, N) — a distribution over N experts
topk_probs, topk_idx = topk(router_probs, k=2, dim=-1)      # keep only the top 2
topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalize the kept 2 to sum to 1

expert_outputs = stack([expert(x) for expert in experts], dim=2)   # every expert, every token (see note below)
mask = zeros_like(router_probs); mask.scatter_(-1, topk_idx, topk_probs)  # renormalized weight at the chosen 2 slots, 0 elsewhere

y = (expert_outputs * mask.unsqueeze(-1)).sum(dim=2)        # = Σ_{i in top-2} weight_i * expert_i(x)
```

That last line is the actual mixture: multiplying by `mask` and summing zeroes out every non-selected expert's contribution, so the result is a weighted combination of exactly the top-2 experts' outputs for that token. Note the code runs **every** expert on **every** token (`stack([expert(x) for expert in experts], ...)`) and only masks afterward — the reference implementation trades compute efficiency for clarity; a real production MoE kernel instead *routes* tokens to experts (so unrouted tokens never touch an expert they weren't assigned to) rather than computing then discarding.

**The load-balancing auxiliary loss:**

```
importance_n = mean over all (batch, position) of router_probs[..., n]   # fully differentiable — a soft signal
load_n = (Σ mask[..., n] over batch & position) / (Σ mask over everything)  # depends on the discrete top-k choice

aux_loss = N * Σ_n importance_n * load_n
```

`topk` itself has no gradient (it's a discrete selection), so `load_n` alone couldn't be optimized directly. Multiplying it by the fully-differentiable `importance_n` gives the auxiliary loss a gradient path back into the router's weights — pushing `router_probs` to spread more evenly across experts is what actually shows up as `∂aux_loss/∂θ`, even though the *value* of the loss depends on which experts the current (non-differentiable) top-k happened to pick.

## Low-rank adapters — LoRA (Project 21)

The entire mechanism is one class:

```
h = base(x) + (α / r) * (x @ A.T) @ B.T
```

where `A` has shape `(r, in_features)`, `B` has shape `(out_features, r)`, and `r` (the *rank*) is chosen far smaller than `min(in_features, out_features)` — typically 4–64 versus a `d_model` in the hundreds or thousands. `B @ A` (composed, `(out_features, in_features)`) is a rank-≤-`r` matrix: the update to the frozen base weight is constrained to live in a small `r`-dimensional subspace, which is the entire reason LoRA needs so few trainable parameters compared to fine-tuning the full `(out_features, in_features)` weight matrix.

Two initialization details matter: `A` gets small random values (`nn.init.normal_(A, std=0.02)`) but `B` starts at exact zero (`nn.init.zeros_(B)`). Since the adapter's contribution is `B @ A`, and matrix-multiplying anything by a zero matrix gives zero, the adapter computes *no* correction at all at step 0 (`h = base(x) + 0`) — training starts from the exact behavior of the frozen base model and only gradually learns a deviation from it. `base.weight.requires_grad_(False)` freezes the base weight itself, but gradients still flow *through* `base(x)` during backprop (need it for any earlier layers) — only the base weight's own `.grad` is never populated, so `AdamW` (constructed over just the LoRA `A`/`B` parameters) never touches it.

## Preference-optimization losses — DPO, KTO, ORPO, SimPO (Project 24)

All four losses share one building block: a sequence's log-probability under a model, computed via the shift-by-one / `log_softmax` / `gather` / mask / sum idiom from [`02_tensors-and-autograd.md`](02_tensors-and-autograd.md) — call it `logp(y | x)` below for brevity ("the model's implicit reward" for response `y` given prompt `x`).

**DPO.** Compare the policy's preference for the chosen response over the rejected one, *relative to* how strongly the frozen reference model already preferred it:

```
π_logratio  = logp_policy(chosen) - logp_policy(rejected)
ref_logratio = logp_ref(chosen) - logp_ref(rejected)
logits = β * (π_logratio - ref_logratio)
loss = -logsigmoid(logits).mean()
```

The intuition: if the policy is shifting *more* toward preferring `chosen` over `rejected` than the reference model already did, `logits` is large and positive, `sigmoid(logits) ≈ 1`, and the loss is small — moving in the right direction is rewarded. `β` controls how sharply the loss responds to that shift (and, implicitly, how far the policy is allowed to drift from the reference before the loss saturates).

**KTO.** No paired chosen/rejected examples required — just single responses individually labeled desirable or undesirable, each compared against a running KL-divergence estimate (`kl_baseline`) instead of a paired reference logratio:

```
logratio = logp_policy(y) - logp_ref(y)
if desirable:   loss = weight * (1 - sigmoid(β * (logratio - kl_baseline)))
if undesirable: loss = weight * (1 - sigmoid(β * (kl_baseline - logratio)))
```

The sign flip between the two branches is the whole trick: a desirable example's loss shrinks as its implicit reward (`logratio`) rises *above* the baseline, while an undesirable example's loss shrinks as its implicit reward falls *below* the baseline — one shared formula, pointed in opposite directions depending on the label.

**ORPO.** No reference model at all — instead of comparing to a KL baseline, it compares the **odds** of the policy generating the chosen vs. rejected response, alongside an ordinary supervised (SFT) term:

```
odds(p) = p / (1 - p)                     # standard odds, from a probability p
log_odds = log(odds(logp_chosen)) - log(odds(logp_rejected))
         = [logp_chosen - log(1 - exp(logp_chosen))] - [logp_rejected - log(1 - exp(logp_rejected))]

loss = -logp_chosen.mean() + λ * ( -logsigmoid(log_odds).mean() )
       └── plain NLL on chosen ──┘   └── odds-ratio preference term ──┘
```

**A genuine numerical landmine**, transcribed directly from the paper's formula with no stabilized rewrite: `log(1 - exp(logp))` breaks down as `logp → 0` (i.e. as the model becomes very confident) — `exp(logp) → 1`, so `1 - exp(logp) → 0`, and its `log` heads to `-∞`. §1.3's underflow warning is exactly this case; don't copy this line into new code without a `log1mexp`-style rewrite that handles `logp` near zero safely.

**SimPO.** Reference-free like ORPO, but length-normalizes the log-probabilities instead of using an odds ratio, and adds an explicit target margin `γ`:

```
chosen_avg   = logp_chosen / len(chosen)
rejected_avg = logp_rejected / len(rejected)
logits = β * (chosen_avg - rejected_avg) - γ
loss = -logsigmoid(logits).mean()
```

Dividing by sequence length before comparing is the point: without it, a longer `chosen` response accumulates more (negative) log-probability just from having more tokens, which would bias the comparison against longer-but-equally-good responses. `γ` sets a minimum reward gap the policy is pushed to satisfy, not just "better than," but "better by at least `γ`."

## Policy gradients and RLHF (Project 23)

The reward model is a linear probe on the final hidden state: `score = w @ h_last + b` — nothing more exotic than an `nn.Linear` down-projecting to a single scalar. The training loop implements a REINFORCE-style policy gradient with a group-relative baseline (this is the "GRPO" in the project title — **G**roup **R**elative **P**olicy **O**ptimization):

```
candidates = [policy.generate(prompt) for _ in range(G)]     # G samples for the same prompt
rewards = reward_model.score(prompt, candidates)              # one scalar reward per candidate
advantages = rewards - rewards.mean(dim=1, keepdim=True)       # subtract the group's own mean reward

logps = policy.logprob(prompt, candidates)
loss = -(advantages.detach() * logps).mean() + β * kl_to_reference(...)
```

This is the score-function estimator for `∇E[reward]`: `∇_θ E[R] = E[R · ∇_θ log π_θ(action)]`, approximated here by sampling `G` actions and averaging. Subtracting a baseline (here, the group's own mean reward) from `R` before multiplying is a standard variance-reduction trick — it doesn't change the expected gradient (a baseline that doesn't depend on the sampled action is provably unbiased to subtract) but it does shrink the *variance* of the estimate, which is what makes this trainable at all with a small `G`. `.detach()` on `advantages` is what enforces "baseline that doesn't depend on the action" mathematically: if gradient were allowed to flow back through the reward/advantage computation itself, the estimator would no longer be the plain policy gradient — only the `logps` term is meant to carry gradient. (The original GRPO paper also divides the centered reward by the group's standard deviation; this book's version only subtracts the mean.) The `β * kl_to_reference(...)` term is a penalty keeping the policy from drifting arbitrarily far from a frozen reference model, the same KL-constraint idea underlying DPO above, just enforced by an explicit added penalty rather than folded into a closed-form loss.

## Quantization (Project 27)

Symmetric integer quantization, precisely as implemented:

```
scale = max(|w|) / 127                              # 127 = 2^(8-1) - 1, the largest signed int8 magnitude used
q = clamp(round(w / scale), -127, 127)               # round to nearest representable level, then clip
w_hat = q.float() * scale                            # dequantize: this is what the model actually computes with
```

`scale` is chosen so the single largest-magnitude weight maps to exactly `±127` — every other weight maps to `round(w / scale)`, an integer with rounding error of at most `±scale/2`. That rounding error is quantization's entire cost: `w_hat` is not `w`, and the gap between them (`(w - w_hat)`) is what Project 27 measures layer-by-layer via `mse = mean((y_fp - y_q)²)`. Crucially, the code above always converts back to `float` (`w_hat = q.float() * scale`) before the matmul (`x @ w_hat.t()`) — it measures the *rounding error* quantization would introduce, but never actually performs a real low-bit integer matmul.

The INT4 group-wise variant applies the identical formula to small chunks (`group_size=64` elements) independently rather than to the whole tensor at once (`scale = max(|chunk|) / 7`, `q = clamp(round(chunk/scale), -8, 7)` — 4 bits sign to `-8..7`), so one outlier weight only blows up the scale (and thus the rounding error) for its own 64-element group instead of the entire layer.

## Vision layers (Project 29)

`nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)` with `stride == kernel_size` is mathematically identical to: cut the image into non-overlapping `patch_size × patch_size` squares, flatten each square into a vector, and apply one shared linear projection to every patch-vector. A convolution normally *slides* its kernel across overlapping positions; setting the stride equal to the kernel size makes every application non-overlapping, which is exactly "chop into patches, then project" — the `nn.Conv2d` is just a convenient, hardware-optimized way to express that operation, followed by `.flatten(2).transpose(1, 2)` to turn the resulting `(B, embed_dim, H', W')` grid into a `(B, H'×W', embed_dim)` token sequence you can concatenate with text embeddings.

## Custom recurrent / state-space blocks — Mamba and RWKV (Project 30)

**The selective SSM (`S6Block`).** The continuous-time model being discretized is a linear ODE, `ḣ(t) = A h(t) + B x(t)`, `y(t) = C h(t)` — a hidden state `h` that decays/evolves according to `A` and is driven by input `x` through `B`, read out through `C`. Turned into a discrete-time recurrence with step size `Δ` (called `dt` in the code):

```
A_bar = exp(Δ * A)                    # exact, because A is diagonal here (matrix exp of a diagonal matrix
                                       # is just the elementwise exp of its diagonal entries)
B_bar = Δ * B                          # a first-order (Euler) approximation of the exact zero-order-hold
                                       # formula B̄ = A⁻¹(exp(ΔA) - I)B, valid when Δ is small

h_t = A_bar * h_{t-1} + B_bar * x_t     # the recurrence, run one timestep at a time in a Python for-loop
y_t = (C_t * h_t).sum(-1) + D * x_t     # readout, plus a per-channel skip connection via D
```

`A` is stored as `A_log = log(A)` and reconstructed as `-exp(A_log)`, guaranteeing it's always negative — a negative `A` means `exp(Δ·A) ∈ (0, 1)`, i.e. the hidden state *decays* rather than explodes, which is what makes the recurrence stable regardless of what gets learned. `B`, `C`, and `Δ` are all produced from the current input `x` by a shared linear projection (`softplus` keeps `Δ > 0`, since a negative step size wouldn't make sense) — this input-dependence is the "selective" in *selective* state-space model: unlike a fixed linear recurrence, the effective dynamics change token-by-token based on content. Notice the recurrence runs as an explicit Python `for t in range(L)` loop — real Mamba implementations use a parallel-scan algorithm to vectorize this same recurrence across the whole sequence at once; this book's version is intentionally the sequential, easy-to-read reference, not the fast one.

**The RWKV time-mix block.** A different, simpler recurrence — a running weighted average, decaying exponentially over time:

```
decay = exp(-softplus(w_raw))                        # a fixed, learned per-channel decay rate in (0, 1)

num_t = decay * num_{t-1} + exp(k_t) * v_t
den_t = decay * den_{t-1} + exp(k_t)
wkv_t = num_t / den_t
out_t = sigmoid(r_t) * wkv_t
```

`num_t / den_t` is a running, exponentially-decayed weighted average of the value vectors `v_t`, where `exp(k_t)` acts as each timestep's (unnormalized) weight — structurally similar to an attention-weighted average, but computed as a cheap running accumulator instead of an all-pairs softmax over the whole sequence. `sigmoid(r_t)` ("receptance") gates how much of that running average passes through at each step. This is a simplified version of RWKV's actual WKV recurrence — real implementations track a running maximum (the same online-softmax stabilization idea from Project 8) to avoid `exp(k_t)` overflowing, and include an extra "bonus" term giving the *current* token special weight; neither appears in this reference version.

## Long-context RoPE variants — Position Interpolation, NTK-aware, YaRN (Project 16)

Building on Project 7's base RoPE (which rotates each consecutive pair of dimensions `(x_{2i}, x_{2i+1})` by an angle `m · θ_i`, where `m` is the token's absolute position and `θ_i = base^{-2i/d}` is a per-pair frequency that gets slower as `i` grows), all three extension methods work by changing `θ_i` — or effectively the position `m` — without touching the rotation math itself:

- **Position Interpolation (PI):** compresses positions by the context-length ratio `k = target_len / train_len` before computing angles (`angle = (m/k) * θ_i`) — every position is squeezed into the range the model originally saw, at the cost of *resolution* between adjacent positions.
- **NTK-aware scaling:** instead of touching positions, stretches the frequency `base` itself: `new_base = base * k^(d/(d-2))`. This raises `base` just enough that the *slowest*-rotating dimension pairs (large `i`, most affected by never completing a full rotation during training) get compressed the most, while the *fastest*-rotating pairs (small `i`) are left almost untouched — those already complete many full rotations within the training length and extrapolate acceptably on their own.
- **YaRN:** makes that same "leave fast dimensions alone, compress slow ones" idea explicit and smooth. For each dimension pair, count how many full rotations it completes over the training length (`rotations = θ_i * train_len / 2π`), then build a ramp `mask = clamp((rotations - α)/(β - α), 0, 1)` and blend: `final_θ_i = θ_i * (mask + (1-mask)/k)`. Where `mask ≈ 1` (fast, well-observed dimensions), the frequency is left unchanged; where `mask ≈ 0` (slow, under-observed dimensions), it's fully compressed by `1/k`, same as PI — everything in between gets a smooth blend. YaRN also divides attention logits by a temperature `1 + 0.1·ln(k)` before the softmax: longer contexts otherwise tend to produce lower-entropy (peakier) attention distributions than the model was trained on, and this extra softening compensates.

## Contrastive / retrieval losses — InfoNCE (Project 28)

```
sim = (query_embs @ passage_embs.T) / τ         # all-pairs similarity, one row per query
labels = arange(batch_size)                      # query i's positive is passage i, by construction
loss = cross_entropy(sim, labels)
```

This is exactly the softmax/cross-entropy machinery from [`01_math-foundations.md`](01_math-foundations.md) §1.5, repurposed: instead of "the correct vocabulary token," the "correct class" for row `i` is passage `i` — its known-matching passage, guaranteed to sit on the diagonal because the batch was constructed with matched query/passage pairs. Every *other* passage in the batch acts as an implicit negative example, with no separate negative-sampling step required. `τ` (temperature) rescales the similarity scores before the softmax — a smaller `τ` sharpens the distribution, making the loss punish near-miss negatives more aggressively.

## Module-compliance / introspection probes (Project 33)

No new formula — this project reuses the RMS calculation already introduced for RMSNorm (`sqrt(mean(x²))`, [`README.md`](README.md)'s "what you don't need to know" table, Project 7) as a runtime *sanity probe* rather than a layer: computing the RMS of a module's actual output on a fixed reference input and checking it falls in a plausible range (e.g. `0.1` to `10.0`) is a cheap way to catch a module that's structurally compatible (same input/output shape) but behaviorally different (e.g. a `LayerNorm` silently substituted for a `RMSNorm`) — something a purely static check (comparing declared metadata, or even comparing `type(module).__name__`) can't catch on its own.

---

Back to the [main overview](README.md).
