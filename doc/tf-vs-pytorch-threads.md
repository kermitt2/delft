# TensorFlow vs PyTorch: thread-pool behaviour in DeLFT

**The one-line finding: TensorFlow ignores `OMP_NUM_THREADS` entirely; PyTorch
obeys it. So pinning `OMP_NUM_THREADS=1` — a common deployment reflex, and what
the production image was shipping — handicaps the PyTorch backend alone.**

That single setting is what made PyTorch look slower than TF in production.

---

## 1. The measurement

GROBID-shaped work: one sequence per call, ten documents in flight, header model.

| | `OMP_NUM_THREADS=1` | backend default |
|---|---|---|
| PyTorch | 3.47 s | **1.99 s** |
| TensorFlow | 9.91 s | 9.89 s |

- **TF: flat.** 9.91 vs 9.89 s is noise. Its Eigen intra-op pool sizes itself from
  `intra_op_parallelism_threads` / `inter_op_parallelism_threads` and never
  consults OMP — unless it was built against oneDNN
  (`TF_ENABLE_ONEDNN_OPTS=1`), which these images are not.
- **PyTorch: 43% slower at one thread.**
- Note the absolute figures: even hobbled to a single thread, PyTorch still tags
  header **2.8× faster** than TF.

**Important caveat about "TF cannot be tuned":** that conclusion was reached from
the wrong knob. Flatness under `OMP_NUM_THREADS` says nothing about whether TF's
intra-op parallelism matters — it only says OMP is not the way to reach it. There
is a second confound: DeLFT's `configure_threads()` calls the TF setters
explicitly, and an explicit call beats the environment. If a TF thread sweep moves
nothing, that is evidence DeLFT is overriding it, **not** evidence that intra-op
parallelism is irrelevant to TF.

---

## 2. End-to-end, same box — the pin *is* the whole story

| config | PyTorch | TF | verdict |
|---|---|---|---|
| `OMP_NUM_THREADS=1` | 1.12 s/doc | 1.15 s/doc | **level** — reproduces the production picture |
| unpinned | — | — | **PyTorch 1.7× faster**, on 30% less CPU (688 vs 984 CPU-s) |

The pinned row reproduces the production observation (PyTorch roughly level with
TF, slightly behind) from an independent direction. The backend was never the
cause; the pin was.

---

## 3. How many threads PyTorch actually wants: 4

CPU sweep, `grvingt-25`, medium corpus, 1 000 docs, PyTorch backend.
Cells are `wall_s (doc/s) cpu_s cores_used`.

| | thr=1 | thr=4 | thr=8 |
|---|---|---|---|
| cpus=10 conc=10 | 1391 (0.72) 7096 5.1 | 1232 (0.81) 9643 7.8 | **2165 (0.46) 19370 8.9** |
| cpus=10 conc=5 | 1041 (0.96) 3911 3.8 | **926 (1.08) 5019 5.4** | 1167 (0.86) 9343 8.0 |
| cpus=10 conc=3 | 1458 (0.69) 3859 2.6 | 1340 (0.75) 4693 3.5 | 1384 (0.72) 6782 4.9 |
| cpus=20 conc=20 | 1403 (0.71) 7420 5.3 | 898 (1.11) 9220 10.3 | 913 (1.10) 11661 12.8 |
| cpus=20 conc=10 | 811 (1.23) 4070 5.0 | **597 (1.67) 5235 8.8** | 586 (1.71) 6434 11.0 |
| cpus=20 conc=5 | 1043 (0.96) 4048 3.9 | 909 (1.10) 5241 5.8 | 888 (1.13) 6237 7.0 |

Reading it:

- **4 is the plateau at every container width.** ~35% faster than 1 thread at
  `conc = cores/2`, and the same wall time as 8 threads for ~23% less CPU.
- **`thr=1` leaves the box half idle** — 5 of 20 cores at the best concurrency.
- **`thr=8` at `conc = cpus` is pathological.** 10 cores / thr=8 / conc=10 means
  80 threads on 10 cores: the worst row in the file at 2 165 s and 19 370 CPU-s —
  **2.7× the CPU of thr=1 for a slower result.**
- On 20 cores, thr=4 → thr=8 buys 2% wall time for +23% CPU seconds. Not worth it.
- Best throughput per core is `cpus=20 thr=4 conc=10`: 1.67 doc/s at 5 235 CPU-s —
  essentially the same CPU budget as thr=1 (4 070) for 1.36× the throughput.

**On GPU, threads stop mattering**: 568 / 569 / 580 s at thr=1 / 2 / 4, because
the DL work isn't on CPU. What's left (pdfalto, Java, CRF decode) is what's being
timed.

---

## 4. Threads interact with concurrency — and the rule is inverted per backend

Threads cannot be tuned independently of `--n`; the pathological cell above is a
threads × concurrency interaction, not a threads problem.

| | cost per doc as concurrency rises | rule |
|---|---|---|
| **PyTorch** | **doubles** — 9 → 21 CPU-s/doc from `--n 5` to `--n 10` | `--n` = **cores / 2** |
| **TF 0.9.1** | **flat** — 20.5 / 20.8 / 21.2 at `--n` 3 / 5 / 10 | `--n` = **cores** |

PyTorch requests contend once the box is full, so it needs room. TF never
contends, but also cannot fill the box — even at `--n 10` it occupies only 6.3 of
10 cores, so it is concurrency-starved rather than core-starved.

---

## 5. Don't size the pool from `os.cpu_count()`

Pinning the intra-op pool to `os.cpu_count()` — which counts **hyperthreads** —
measured **~70% slower** per-sequence tagging on a 12-core/24-thread Xeon:
**2.57 s vs 1.53 s** over 200 affiliation-address sequences, against simply
leaving torch's own physical-core default in place.

Consequently DeLFT's `configure_threads()`:

- only ever **raises** a clamped pool, never lowers or oversubscribes one;
- **steps aside entirely** when `OMP_NUM_THREADS` / `MKL_NUM_THREADS` are set —
  the operator has already sized it;
- caps at the **cgroup quota** (`--cpus`, v2 `cpu.max` / v1 `cfs_quota_us`) and at
  the **CPU affinity mask** (`--cpuset-cpus`).

Both caps are needed and are not interchangeable: `--cpus` sets a CFS quota and
leaves the affinity mask untouched, while `--cpuset-cpus` narrows the mask and
leaves `cpu.max` at `max`. Torch reads neither, so a container pinned to 4 cores
on a 32-core host would otherwise build a 32-thread pool and spend each inference
in OpenMP barriers waiting on threads the scheduler will never run. Verified under
`taskset -c 0-3` on a 24-core box: 12 threads before, 4 after.

---

## 6. What was changed in the image

`Dockerfile.delft` on `grobidOrg/grobid`, branch `feature/delft-1.0.0-pytorch`
(commit `5ddd748db`) now sets the pool explicitly:

```dockerfile
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4
```

replacing the previous block, which deliberately left both unset and deferred to
DeLFT's `configure_threads()` — which sizes the pool to the whole cpuset, and that
is exactly the oversubscription the sweep exposed.

- **Why `ENV` and not `torch.set_num_threads()`**: OpenMP reads the variable when
  the native libraries load, i.e. at `import torch`. Setting it from Python
  afterwards is too late. With `ENV`, the pool is 4 regardless of how DeLFT
  evolves — and it stays overridable per run with
  `docker run -e OMP_NUM_THREADS=N`, which is how the sweeps set their thread
  axis.
- **Interop pool**: with the env set, `configure_threads()` returns early and never
  calls `set_num_interop_threads`, so torch's default interop pool stands. That is
  the configuration the sweep measured, so the numbers transfer as-is.
- **TF images untouched** — Eigen doesn't read `OMP_NUM_THREADS`, and this is
  `Dockerfile.delft` on the pytorch branch regardless.

Verification after a build:

```bash
docker run --rm <image> python3 -c "import torch, delft; print(torch.get_num_threads())"
# must print 4
```
