# 3. Intermediate: systems and training internals (required by roughly Projects 5–17)

Math background for this section: [`01_math-foundations.md`](01_math-foundations.md) §1.1 and §1.6.

---

## Module introspection

- `model.named_modules()`, `model.modules()`, `model.named_parameters()`, and the non-obvious `module.parameters(recurse=False)` (a module's *own* parameters, not its children's) — used to selectively apply weight decay, custom init, or per-layer instrumentation by `isinstance` type-checking (`isinstance(m, nn.Linear)`, `isinstance(m, nn.LayerNorm)`, etc.)
- `id(param)`-based deduplication — required wherever you walk the module tree and a weight-tied parameter would otherwise get counted or classified twice (the tied-weight gradient-accumulation point from [`02_tensors-and-autograd.md`](02_tensors-and-autograd.md) has a structural twin here: the *same object* appearing twice in a tree walk, not just twice in a graph)

## Checkpointing

- `model.state_dict()` / `load_state_dict()`, `torch.save` / `torch.load`

A `state_dict` is just an ordered dict mapping parameter/buffer names to tensors — no math beyond "here are the numbers, here is what to call them."

## Hooks

- `module.register_forward_hook(fn)` with signature `fn(module, input, output)` — used for activation-statistics logging (Project 11); `.detach()` before computing stats inside a hook, so logging doesn't extend the autograd graph

The statistics themselves are the plain ones: `.mean()` = `(1/n)Σxᵢ`, `.std()` = `sqrt((1/n)Σ(xᵢ-mean)²)`, plus `.min()`/`.max()`/`.abs().max()`. What makes this a *diagnostic* rather than just descriptive stats is what "healthy" ranges look like for activations in a transformer (roughly order-1 magnitude, not collapsing toward 0 or blowing up toward `inf`/`nan`) — that judgment comes from experience, not a formula, which is why Project 11 is about learning to read these numbers rather than deriving them.

## Mixed precision

- `torch.cuda.amp.GradScaler()` and the `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()` pattern, plus reading `scaler.get_scale()` to detect a skipped step (overflow)
- `dtype=torch.float16` / `torch.bfloat16` passed explicitly to tensor constructors
- Worth knowing going in: real mixed-precision training normally pairs `GradScaler` with `torch.autocast(...)` wrapping the forward pass — **this repo never actually shows `autocast`**, only the `GradScaler` overflow-bookkeeping half. Don't expect to find a canonical autocast example here; bring that knowledge with you if you want the full picture.

**Why this machinery exists, numerically.** A 32-bit float (`float32`) spends 8 bits on the exponent and 23 on the mantissa (precision). A 16-bit float (`float16`) spends only 5 bits on the exponent — its representable range collapses to roughly `6×10⁻⁵` to `65504`. Gradients late in training are often smaller than `6×10⁻⁵` in `float16` and simply **underflow to exact zero**, silently killing learning in the affected parameters. `bfloat16` sidesteps this specific problem by keeping `float32`'s 8 exponent bits (same range, so no underflow-to-zero) while giving up more mantissa bits (so it's less *precise*, but doesn't lose small values entirely) — which is why newer hardware/training recipes increasingly prefer `bfloat16` over `float16` and skip `GradScaler` altogether.

**Loss scaling, precisely.** For `float16` specifically, `GradScaler` counteracts underflow by scaling the loss up by a factor `S` before `.backward()`: since gradients scale linearly with the loss (chain rule, §1.2), every gradient in the graph comes out `S×` larger — large enough to clear the `float16` underflow floor — and gets divided back down by `S` before the actual optimizer step, so the *update* is mathematically the same as an unscaled `float32` computation would have produced. `S` is adjusted dynamically: if a step's gradients contain an `inf`/`nan` (overflow — the risk of scaling *up*), that step is skipped and `S` is shrunk; after enough consecutive clean steps, `S` is grown again. `scaler.get_scale()` reading a smaller value than the previous step is exactly how you detect a skipped step from the outside.

## Device and CUDA management

- `device=` kwargs on tensor constructors, `.to(device)`, `torch.cuda.is_available()`
- `torch.cuda.synchronize()` and *why* it's needed before timing anything (CUDA calls are asynchronous by default)
- `torch.cuda.max_memory_allocated()`, `reset_peak_memory_stats()`, `empty_cache()`, and catching `torch.cuda.OutOfMemoryError` — shown in Project 8's step files (not its `build.py`) for measuring FlashAttention's real memory advantage

**Why `synchronize()` matters for timing.** GPU kernels are launched asynchronously — the CPU issues the instruction and immediately moves on to the next line of Python, without waiting for the GPU to actually finish. `time.perf_counter()` calls placed around GPU work without a `synchronize()` in between are measuring "how long it took the CPU to *queue up* the work," not "how long the GPU took to *do* it" — the two can differ by orders of magnitude, which is why every honest GPU benchmark in this space brackets the timed region with `torch.cuda.synchronize()`.

## Batched attention tensor choreography

- The `[B, H, T, d_head]` shape convention and `transpose(1, 2)` to split/merge the head dimension — this is assumed fluent by Project 5 and never re-taught
- Sampling and lookup ops: `torch.multinomial`, `torch.argmax`, `torch.topk`, `.gather()` for per-row probability lookups — dense in Projects 14 (speculative decoding) and 22/24/25 (evaluation, DPO, reasoning)

**Scaled dot-product attention, the formula every one of these shape manipulations is building toward:**

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_head ) V
```

`Q Kᵀ` (shape `(T, T)`, or `(B, H, T, T)` batched) is a matrix of raw similarity scores between every query position and every key position (§1.1's matmul rule, applied to the trailing two dimensions while `B` and `H` broadcast along for the ride — this is *why* the `[B, H, T, d_head]` layout exists: reshape once into "batch of matrices" form, then every subsequent operation is an ordinary 2-D matmul or softmax applied per matrix). The `1/√d_head` scaling exists for a specific numerical reason: if `Q` and `K`'s entries are roughly independent with unit variance, a dot product of `d_head` such terms has variance that grows linearly with `d_head`, so its typical magnitude grows with `√d_head`. Left unscaled, larger head dimensions would push the pre-softmax scores to extremes, saturating the softmax (nearly all probability mass on one position, everywhere) regardless of whether that's actually where attention *should* concentrate. Dividing by `√d_head` cancels that growth, keeping the score distribution's scale roughly constant no matter how large `d_head` is — Project 4's BREAK IT experiment (ablating exactly this term) is a direct demonstration.

## Distributed training

(concentrated almost entirely in Project 12, which is the single most PyTorch-systems-heavy project in the book)

- `torch.distributed`: `init_process_group(backend=...)`, `destroy_process_group()`, `all_reduce`, `all_gather`, `reduce_scatter_tensor`, `ReduceOp.SUM`
- `torch.multiprocessing.spawn(fn, args=..., nprocs=...)` and its rank-injection convention (each spawned process calls `fn(rank, *args)`)
- `torch.distributed.fsdp.FullyShardedDataParallel` (FSDP) and `torch.distributed.fsdp.wrap.transformer_auto_wrap_policy` — wrapping each transformer block as its own FSDP unit rather than the whole model at once, and why that controls the granularity of the all-gather/reduce-scatter traffic
- Process groups, rank, world size, and backend choice (`"gloo"` for the CPU-friendly single-box proxy in this repo vs. `"nccl"` for real multi-GPU, which is never shown)

**The three collectives, precisely** (`N` = world size, one tensor per rank unless noted):

```
all_reduce(x, op=SUM)      every rank ends up with the same value: Σᵢ xᵢ
                            (gradient averaging = all_reduce(SUM) then divide by N)

all_gather(pieces, shard)  every rank ends up with the full list [shard₀, ..., shard_{N-1}]
                            (reconstructing a full sharded parameter from N per-rank pieces)

reduce_scatter(out, x)     x is summed across ranks (like all_reduce), but instead of every
                            rank getting the full sum, rank i gets only the i-th slice of it
                            (distributing a summed gradient back out as N per-rank shards)
```

`all_gather` and `reduce_scatter` are each other's mirror image: one takes `N` partial pieces and gives everyone the whole, the other takes one whole (implicitly, via the sum) and gives everyone back only their own piece. FSDP's per-step pattern is exactly this pair used back-to-back: `all_gather` each layer's full weight just-in-time for its forward/backward pass, then `reduce_scatter` the resulting gradient back down to per-rank shards immediately afterward, so no rank ever holds a second full copy of the optimizer state for a layer it isn't actively computing. Wrapping each `TransformerBlock` as its own FSDP unit (rather than the whole model as one unit) is what makes this per-layer just-in-time gather/scatter possible instead of gathering the entire model's parameters at once.

---

Next: [`04_advanced-topics.md`](04_advanced-topics.md) — MoE, LoRA, preference optimization, RLHF, quantization, and non-transformer architectures (Tier 3, Projects 18–35).
