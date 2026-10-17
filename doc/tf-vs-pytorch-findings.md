# DeLFT: TensorFlow vs PyTorch — performance findings

Summary of the benchmarking done across the `feature/improve-pytorch-performances`
work (DeLFT 0.4.x/TF → 1.0.x/PyTorch), including the GROBID-level sweeps on
Grid5000.

**Headline: PyTorch is faster than TensorFlow at every level we measured — roughly
2× end-to-end throughput on heavy PDFs at less than half the CPU cost per
document, and 6× on isolated per-model tagging at batch 1.** The initial
production reading that said the opposite was a configuration artefact, not a
property of the backend.

---

## 1. The conclusion changed three times — worth knowing why

This matters more than any single number, because the first two readings were
wrong and were acted upon.

| Stage | Reading | What was actually going on |
|---|---|---|
| Production, first look | PyTorch **slower** — 3.26 s/doc vs TF's 2.10 s/doc | Compared against a TF baseline whose node, corpus and model set were never pinned down |
| Isolated benchmarks | PyTorch **6× faster** | True, but per-call tagging only — it did not explain end-to-end throughput |
| Controlled sweep (final) | PyTorch **2× faster**, at 43% of the CPU | Same cluster, same corpus, same GROBID build, each backend at its own optimum |

The three confounders that produced the false negative:

1. **`OMP_NUM_THREADS=1` was pinned in the production image.** PyTorch obeys it;
   TF does not (see §4). That pin alone cost ~43%.
2. **Node variance.** ~23% between the Broadwell and Skylake machines on identical
   work. The node the 2.10 s/doc TF baseline ran on was never identified.
3. **Different model sets.** An early A/B used restored `BidLSTM_CRF_FEATURES`
   `.hdf5` weights, while the 2.10 s/doc production figure came from **ChainCRF**
   models — a different decode path entirely. That comparison was never valid.

---

## 2. Final controlled comparison — GROBID, large corpus, 10 cores, threads=4

Same cluster (`grvingt`), same CPU model, same corpus, 10 000 documents each.
Each backend at its own best concurrency.

| | best config | doc/s | docs/hour | CPU-s per doc |
|---|---|---|---|---|
| **PyTorch** | `--n 5` | **0.61** | **2 200** | **9.0** |
| **TF 0.9.1** | `--n 10` | 0.30 | 1 060 | 21.2 |

**2× the throughput on 43% of the CPU budget.** Failure counts are the same
(~65–70 per 10 000), so nothing is being traded for reliability.

TF's ~21 CPU-s/doc is its irreducible price on these documents; PyTorch's is 9.
Even if TF could fill all 10 cores it would reach only ~0.47 doc/s — still below
PyTorch's 0.61.

---

## 3. The tuning rule is *inverted* between the two backends

This is the most operationally useful finding, and it is easy to get backwards.

| | cost per doc as concurrency rises | rule |
|---|---|---|
| **PyTorch** | **doubles** — 9 → 21 CPU-s/doc from `--n 5` to `--n 10` | `--n` = **cores / 2** |
| **TF 0.9.1** | **flat** — 20.5 / 20.8 / 21.2 at `--n` 3 / 5 / 10 | `--n` = **cores** |

PyTorch requests fight each other once the box is full, so it must be given room.
TF never contends — but it also cannot fill the box: even at `--n 10` it occupies
only 6.3 of 10 cores, so it is concurrency-starved rather than core-starved.

PyTorch only looks bad when oversubscribed. Most of the original "PyTorch is
slow" impression was exactly this.

**Recommended production config (PyTorch):** `--cpuset-cpus` = 20,
`OMP_NUM_THREADS=4`, `--n 10`.

---

## 4. Thread-pool sensitivity — the root cause of the false negative

Measured on GROBID-shaped work (one sequence per call, ten documents in flight,
header model):

| | `OMP_NUM_THREADS=1` | backend default |
|---|---|---|
| PyTorch | 3.47 s | **1.99 s** |
| TensorFlow | 9.91 s | 9.89 s |

- **TF is completely insensitive to `OMP_NUM_THREADS`** — its Eigen intra-op pool
  never reads it (it uses `intra_op/inter_op_parallelism_threads` instead).
- **PyTorch tags 43% slower when pinned to one thread.** So `OMP_NUM_THREADS=1`,
  a common deployment reflex, handicaps the PyTorch backend *alone*.
- Note that even hobbled to a single thread, PyTorch still tags header **2.8×
  faster** than TF (3.47 s vs 9.91 s).

Two related measurements:

- Pinning the pool to `os.cpu_count()` (i.e. hyperthreads) made per-sequence
  tagging **~70% slower** on a 12-core/24-thread Xeon — 2.57 s vs 1.53 s. DeLFT
  therefore only ever *raises* a clamped pool, never lowers or oversubscribes it.
- `OMP_NUM_THREADS=4` is the plateau at every container width: ~35% faster than 1
  thread at `conc = cores/2`, and the same wall time as 8 threads for ~23% less
  CPU.

---

## 5. Isolated per-model tagging — TF 0.4.6 vs PyTorch branch

Same hardware, same sequences, same harness, best of three, with warmup.
Restored `.hdf5` models, delft 0.4.6 + TF 2.17.1.

| model | TF b=1 | PyTorch b=1 | speedup | TF b=20 | PyTorch b=20 | speedup | label agreement |
|---|---|---|---|---|---|---|---|
| header | 10.22 s | 1.67 s | **6.1×** | 3.71 s | 1.85 s | 2.0× | 99.7% |
| reference-segmenter | 12.48 s | 2.05 s | **6.1×** | — | — | — | 99.9% |
| affiliation-address | 6.22 s | 0.92 s | **6.8×** | 1.66 s | 0.88 s | 1.9× | 96.4% |
| citation | 6.20 s | 0.99 s | **6.3×** | 0.97 s | 0.37 s | 2.6× | 99.1% |
| name-header | 4.71 s | 0.71 s | **6.6×** | — | — | — | 98.7% |
| funding-acknowledgement | failed | 1.13 s | — | — | — | — | — |

**PyTorch is 6.1–6.8× faster at batch 1 and 1.9–2.6× at batch 20. There is no
configuration in which TF wins.** Batch 1 is the regime GROBID actually uses for
the header and reference-segmenter models.

`funding-acknowledgement` failed under TF; unexplained, not chased.

---

## 6. Production run, 10 000 documents

| run | s/doc | doc/s | runtime |
|---|---|---|---|
| before | 3.26 | 0.31 | 32 371 s |
| **after (CPU)** | **2.73** | **0.37** | **27 097 s** |
| after (GPU) | 3.17 | 0.32 | 31 503 s |
| TF baseline (unverified provenance) | 2.10 | 0.48 | 20 807 s |

The perf branch delivered a real **16% end-to-end improvement** (5 274 seconds
saved on 10k documents). An earlier run had shown no change at all — that was a
stale Docker layer, fixed by pinning the delft SHA.

**GPU is 16% slower than CPU here** and is not worth the reservation: `gpu_util`
sits at 7–10%, and batch-1 recurrent inference is launch-latency bound. The CRF
decode is deliberately kept on CPU, which adds a device→host transfer per call.

The residual gap to the 2.10 s/doc figure was never closed — but see §1, that
baseline's node, corpus and model set are unknown, and the controlled sweep in §2
reverses the sign entirely.

---

## 7. Accuracy — not a speed-for-quality trade

The PyTorch port initially *lost* accuracy through a masking regression, since
fixed in `dev`:

- **Character-level masking** (#231, `40781b6`): collapsing per-architecture
  `mask_zero` onto one shared `CharacterEncoder` lost masking for the
  architectures that had it. On GROBID 0.8.2 header-BidLSTM_CRF weights over 20
  documents: **0.5219 → 0.7583 micro F1**, against **0.7724** from the TensorFlow
  implementation. Verified against Keras itself — the masked encoder matches to
  2.8e-9 where the unmasked one differs in the third decimal.
- **Token-level masking** (#232, `6bcab5c`) closed the remainder.

Label agreement between backends is **96–100%** across models with matching
sequence lengths; the residual is independently retrained weights, not a bug.

**One open item worth an eval before declaring the migration free:** on identical
inputs, TF produced 4 888 `biblStruct` entries against PyTorch's 4 667 — **4.5%
more references extracted**. Total TEI volume is close (10.05 vs 9.98 MB), so
neither is degenerate, but the `.pt` and `.hdf5` weights were trained separately
and are not the same model. This is a recall question, independent of everything
measured above.

---

## 8. What changed in DeLFT to get here

Optimisations on `feature/improve-pytorch-performances` (PyTorch-side, cumulative
~32–37% on isolated tagging; header per-sequence tagging trended ~3.76 s → ~2.17 s):

| change | effect |
|---|---|
| TorchScript Viterbi decode replacing pytorch-crf's Python loop | ~2× decode |
| Reduced per-timestep ATen dispatch in the decode | 16.4 → 11.3 ms (457 tokens, 44 tags) |
| numpy decode for narrow batches (crossover at batch ≤ 3) | 1.38× at batch 1 |
| `inference_mode` instead of `no_grad` | 523 → 439 ms |
| Feature-position resolution once per batch, not per token | 260 → 165 ms (27k-token slice) |
| Device resolved once; `Tagger` reused across `tag()` calls | header 3.76 → 3.04 s |
| float32 word-embedding rows (no float64 round trip) | halved memory traffic |
| Thread pool capped at cgroup quota and CPU affinity mask | prevents the container oversubscription above |

Deployment change: `Dockerfile.delft` on `grobidOrg/grobid`
`feature/delft-1.0.0-pytorch` now sets `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4`
explicitly (commit `5ddd748db`), rather than leaving DeLFT to size the pool to the
whole cpuset.

---

## 9. Caveats

- The 2.10 s/doc TF production baseline's **node, corpus and model set are
  unknown**. Do not treat it as a target; §2 is the trustworthy comparison.
- The final TF rows are three configurations at `threads=4`. No TF thread setting
  was found that closes the gap, but the TF thread axis was not swept
  exhaustively — it would need to close a 2× gap *and* flatten 21 CPU-s/doc down
  to 9.
- `--n` above the core count is untested for TF and is the one cell that might
  still help it.
- The GPU finding is specific to this image and corpus. The bottleneck there is
  the non-DL pipeline (pdfalto, Java, CRF decode), not the model.
