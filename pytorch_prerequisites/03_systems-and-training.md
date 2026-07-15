# 3. Intermediate: systems and training internals (required by roughly Projects 5–17)

Math background for this section: [`01_math-foundations.md`](01_math-foundations.md) §1.1 and §1.6. As in the previous file, every snippet is small on purpose — nothing here needs a GPU or more than a fraction of a second to run.

---

## Module introspection

- `model.named_modules()`, `model.modules()`, `model.named_parameters()`, and the non-obvious `module.parameters(recurse=False)` (a module's *own* parameters, not its children's) — used to selectively apply weight decay, custom init, or per-layer instrumentation by `isinstance` type-checking (`isinstance(m, nn.Linear)`, `isinstance(m, nn.LayerNorm)`, etc.)
- `id(param)`-based deduplication — required wherever you walk the module tree and a weight-tied parameter would otherwise get counted or classified twice

```python
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])

m = Toy()
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear):
        print("Linear at:", name)     # blocks.0.fc, blocks.1.fc
```

**Plain `model.parameters()` already deduplicates tied weights for you** — worth confirming, since it's easy to assume you always need to guard against it yourself:

```python
class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 4)
        self.head = nn.Linear(4, 5, bias=False)
        self.head.weight = self.emb.weight

m = Tied()
print(len(list(m.parameters())))   # 1, not 2 -- PyTorch already collapses the shared object
```

Where `id()`-based dedup actually earns its keep is a *different* traversal pattern — the one Project 6 uses: walking every module and asking each one, individually, for only the parameters it *directly* owns (`recurse=False`), in order to classify each parameter by its owning module's type:

```python
seen, deduped = set(), []
for name, mod in m.named_modules():
    for p in mod.parameters(recurse=False):     # each module reports only its OWN parameters
        if id(p) not in seen:
            seen.add(id(p))
            deduped.append((name, p))
        else:
            print(f"skipped a duplicate seen again at {name}")   # prints once, for 'head'

print(len(deduped))   # 1
```

Here the tied weight genuinely gets visited twice — once while asking `emb` for its own parameters, once while asking `head` for its own parameters, since both modules directly hold a reference to the same `Parameter` object. `.parameters()`'s built-in dedup doesn't help you here because you're not calling it; you're doing your own per-module walk, which is exactly the situation `id(param) in seen_params` guards against.

## Checkpointing

- `model.state_dict()` / `load_state_dict()`, `torch.save` / `torch.load`

A `state_dict` is just an ordered dict mapping parameter/buffer names to tensors — no math beyond "here are the numbers, here is what to call them."

## Hooks

- `module.register_forward_hook(fn)` with signature `fn(module, input, output)` — used for activation-statistics logging (Project 11); `.detach()` before computing stats inside a hook, so logging doesn't extend the autograd graph

```python
import torch
import torch.nn as nn

activations = {}
def make_hook(name):
    def hook(module, inputs, output):
        activations[name] = output.detach()    # detach: just logging, not part of the graph
    return hook

lin = nn.Linear(3, 4)
handle = lin.register_forward_hook(make_hook("lin"))
y = lin(torch.randn(2, 3))
print(activations["lin"].shape)    # torch.Size([2, 4]) -- captured without changing lin's actual output
handle.remove()                     # hooks stay attached until removed -- easy to forget
```

The statistics themselves are the plain ones: `.mean()` = `(1/n)Σxᵢ`, `.std()` = `sqrt((1/n)Σ(xᵢ-mean)²)`, plus `.min()`/`.max()`/`.abs().max()`. What makes this a *diagnostic* rather than just descriptive stats is what "healthy" ranges look like for activations in a transformer (roughly order-1 magnitude, not collapsing toward 0 or blowing up toward `inf`/`nan`) — that judgment comes from experience, not a formula, which is why Project 11 is about learning to read these numbers rather than deriving them.

## Mixed precision

- `torch.cuda.amp.GradScaler()` and the `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()` pattern, plus reading `scaler.get_scale()` to detect a skipped step (overflow)
- `dtype=torch.float16` / `torch.bfloat16` passed explicitly to tensor constructors
- Worth knowing going in: real mixed-precision training normally pairs `GradScaler` with `torch.autocast(...)` wrapping the forward pass — **this repo never actually shows `autocast`**, only the `GradScaler` overflow-bookkeeping half. Don't expect to find a canonical autocast example here; bring that knowledge with you if you want the full picture.

**Why this machinery exists, numerically.** A 32-bit float (`float32`) spends 8 bits on the exponent and 23 on the mantissa (precision). A 16-bit float (`float16`) spends only 5 bits on the exponent — its representable range collapses to roughly `6×10⁻⁵` to `65504`. Gradients late in training are often smaller than `6×10⁻⁵` in `float16` and simply **underflow toward zero or lose precision**, silently hurting learning in the affected parameters:

```python
import torch

x = torch.tensor(1e-6, dtype=torch.float16)
print(x.item())    # ~1.0133e-06 -- already imprecise; float16's smallest *normal* value is ~6.1e-5

# GradScaler's fix: scale UP before casting down, so the value sits in float16's precise range,
# then divide back down by the same factor afterward:
big = (torch.tensor(1e-6, dtype=torch.float32) * 65536).to(torch.float16)
print(big.item(), "-> unscaled:", (big.float() / 65536).item())   # recovers ~1e-6 much more precisely
```

`bfloat16` sidesteps this specific problem by keeping `float32`'s 8 exponent bits (same range, so no underflow-to-zero) while giving up more mantissa bits (so it's less *precise*, but doesn't lose small values entirely) — which is why newer hardware/training recipes increasingly prefer `bfloat16` over `float16` and skip `GradScaler` altogether.

**Loss scaling, precisely.** For `float16` specifically, `GradScaler` counteracts underflow by scaling the loss up by a factor `S` before `.backward()`: since gradients scale linearly with the loss (chain rule, §1.2), every gradient in the graph comes out `S×` larger — large enough to clear the `float16` underflow floor — and gets divided back down by `S` before the actual optimizer step, so the *update* is mathematically the same as an unscaled `float32` computation would have produced. `S` is adjusted dynamically: if a step's gradients contain an `inf`/`nan` (overflow — the risk of scaling *up*), that step is skipped and `S` is shrunk; after enough consecutive clean steps, `S` is grown again. `scaler.get_scale()` reading a smaller value than the previous step is exactly how you detect a skipped step from the outside.

## Device and CUDA management

- `device=` kwargs on tensor constructors, `.to(device)`, `torch.cuda.is_available()`
- `torch.cuda.synchronize()` and *why* it's needed before timing anything (CUDA calls are asynchronous by default)
- `torch.cuda.max_memory_allocated()`, `reset_peak_memory_stats()`, `empty_cache()`, and catching `torch.cuda.OutOfMemoryError` — shown in Project 8's step files (not its `build.py`) for measuring FlashAttention's real memory advantage

This guide itself was written and every snippet in it verified on a CPU-only machine — exactly the situation most readers start from:

```python
import torch
print(torch.cuda.is_available())   # False, on a CPU-only machine (no error — just nothing to use)
```

**Why `synchronize()` matters for timing.** GPU kernels are launched asynchronously — the CPU issues the instruction and immediately moves on to the next line of Python, without waiting for the GPU to actually finish. `time.perf_counter()` calls placed around GPU work without a `synchronize()` in between are measuring "how long it took the CPU to *queue up* the work," not "how long the GPU took to *do* it" — the two can differ by orders of magnitude, which is why every honest GPU benchmark in this space brackets the timed region with `torch.cuda.synchronize()`. (There's nothing to demonstrate for this on CPU — the asynchrony this guards against is specifically a GPU behavior.)

## Batched attention tensor choreography

- The `[B, H, T, d_head]` shape convention and `transpose(1, 2)` to split/merge the head dimension — this is assumed fluent by Project 5 and never re-taught
- Sampling and lookup ops: `torch.multinomial`, `torch.argmax`, `torch.topk`, `.gather()` for per-row probability lookups — dense in Projects 14 (speculative decoding) and 22/24/25 (evaluation, DPO, reasoning)

**Scaled dot-product attention, the formula every one of these shape manipulations is building toward:**

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_head ) V
```

```python
import torch
import torch.nn.functional as F
import math

def attention(Q, K, V, scale=True):
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)             # (T, T): every query dotted with every key
    if scale:
        scores = scores / math.sqrt(d)
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights

T, d = 4, 8
Q, K, V = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
out, weights = attention(Q, K, V)
print(out.shape, weights.sum(dim=-1))   # torch.Size([4, 8])  tensor([1., 1., 1., 1.]) -- each row of weights sums to 1
```

`Q Kᵀ` (shape `(T, T)`, or `(B, H, T, T)` batched) is a matrix of raw similarity scores between every query position and every key position (§1.1's matmul rule, applied to the trailing two dimensions while `B` and `H` broadcast along for the ride — this is *why* the `[B, H, T, d_head]` layout exists: reshape once into "batch of matrices" form, then every subsequent operation is an ordinary 2-D matmul or softmax applied per matrix). The `1/√d_head` scaling exists for a specific numerical reason, directly measurable:

```python
import torch, math

for d in (4, 64, 256):
    torch.manual_seed(0)
    q, k = torch.randn(2000, d), torch.randn(2000, d)
    dot = (q * k).sum(dim=-1)                    # 2000 sample dot products at this dimension
    print(f"d={d:4d}  unscaled std={dot.std():.2f}   scaled (÷√d) std={(dot / math.sqrt(d)).std():.2f}")

# d=   4  unscaled std=2.03   scaled std=1.01
# d=  64  unscaled std=8.05   scaled std=1.01
# d= 256  unscaled std=15.84  scaled std=0.99
```

If `Q` and `K`'s entries are roughly independent with unit variance, a dot product of `d_head` such terms has variance that grows linearly with `d_head` — the printed standard deviations above roughly double each time `d` quadruples, exactly the `√d` relationship the formula predicts. Left unscaled, larger head dimensions would push the pre-softmax scores to extremes, saturating the softmax (nearly all probability mass on one position, everywhere) regardless of whether that's actually where attention *should* concentrate. Dividing by `√d_head` keeps the score distribution's scale roughly constant (note the scaled column stays close to `1.0` at every `d` above) no matter how large `d_head` is — Project 4's BREAK IT experiment (ablating exactly this term) is a direct demonstration.

## Distributed training

(concentrated almost entirely in Project 12, which is the single most PyTorch-systems-heavy project in the book)

- `torch.distributed`: `init_process_group(backend=...)`, `destroy_process_group()`, `all_reduce`, `all_gather`, `reduce_scatter_tensor`, `ReduceOp.SUM`
- `torch.multiprocessing.spawn(fn, args=..., nprocs=...)` and its rank-injection convention (each spawned process calls `fn(rank, *args)`)
- `torch.distributed.fsdp.FullyShardedDataParallel` (FSDP) and `torch.distributed.fsdp.wrap.transformer_auto_wrap_policy` — wrapping each transformer block as its own FSDP unit rather than the whole model at once, and why that controls the granularity of the all-gather/reduce-scatter traffic
- Process groups, rank, world size, and backend choice (`"gloo"` for the CPU-friendly single-box proxy in this repo vs. `"nccl"` for real multi-GPU, which is never shown)

**The three collectives, precisely** (`N` = world size, one tensor per rank unless noted). The actual `torch.distributed` API needs multiple processes to run for real, so this is what each one computes, simulated here as plain Python lists standing in for "one tensor per rank" — the arithmetic is identical to the real, multi-process version:

```python
import torch

N = 4
per_rank = [torch.arange(N).float() + rank * 10 for rank in range(N)]
for r, t in enumerate(per_rank):
    print(f"rank {r} starts with: {t}")
# rank 0: [0,1,2,3]   rank 1: [10,11,12,13]   rank 2: [20,21,22,23]   rank 3: [30,31,32,33]

# all_reduce(SUM): every rank ends up holding the SAME full sum
all_reduce_result = sum(per_rank)
print("all_reduce(SUM) ->", all_reduce_result, "(identical on every rank)")
# tensor([60., 64., 68., 72.])

# all_gather: every rank ends up holding the full LIST of every rank's original piece
all_gather_result = torch.stack(per_rank)
print("all_gather -> shape", all_gather_result.shape, "(every rank gets all N pieces)")
# torch.Size([4, 4])

# reduce_scatter: sum like all_reduce, but rank i keeps ONLY element i of the result
print("reduce_scatter -> each rank keeps only its own slice of the sum:")
for r in range(N):
    print(f"  rank {r} keeps: {all_reduce_result[r].item()}")
# rank 0 keeps: 60.0   rank 1 keeps: 64.0   rank 2 keeps: 68.0   rank 3 keeps: 72.0
```

`all_gather` and `reduce_scatter` are each other's mirror image: one takes `N` partial pieces and gives everyone the whole, the other takes one whole (implicitly, via the sum) and gives everyone back only their own piece. FSDP's per-step pattern is exactly this pair used back-to-back: `all_gather` each layer's full weight just-in-time for its forward/backward pass, then `reduce_scatter` the resulting gradient back down to per-rank shards immediately afterward, so no rank ever holds a second full copy of the optimizer state for a layer it isn't actively computing. Wrapping each `TransformerBlock` as its own FSDP unit (rather than the whole model as one unit) is what makes this per-layer just-in-time gather/scatter possible instead of gathering the entire model's parameters at once.

---

## Glossary

- **Hook** — a function PyTorch calls automatically at a specific point (e.g. right after a module's forward pass) without you having to modify that module's code.
- **Checkpoint** — a saved snapshot of a model's (and often optimizer's) state, so training or inference can resume without starting over.
- **Mixed precision** — running most of a model's compute in a lower-precision format (`float16`/`bfloat16`) for speed and memory, while keeping select parts (like gradient accumulation) in `float32` for numerical safety.
- **Overflow / underflow** — a number becoming too large to represent (rounds to `inf`) or too small (rounds to `0`) in a given numeric format.
- **Process group** — the set of parallel processes (e.g. one per GPU) participating together in a distributed computation.
- **Rank** — a process's index within its process group (0, 1, 2, ...); **world size** is the total number of processes.
- **Collective operation** — a communication pattern involving every process in a group at once (as opposed to one process talking to just one other), e.g. all-reduce, all-gather, reduce-scatter.
- **Sharding** — splitting a single large tensor (e.g. a model's weights) into pieces distributed across multiple devices, so no single device needs to hold the whole thing.

Next: [`04_advanced-topics.md`](04_advanced-topics.md) — MoE, LoRA, preference optimization, RLHF, quantization, and non-transformer architectures (Tier 3, Projects 18–35).
