#!/usr/bin/env python
"""
Time the CRF training loss (forward and backward) of pytorch-crf against three rewrites.

The loss delft trains with is ``-torchcrf.CRF.forward(...)`` (``delft/utilities/crf_pytorch.py``).
pytorch-crf loops over the sequence one timestep at a time, twice (the score of the gold path
and the normalizer), and indexes ``emissions[i]`` inside both loops, which makes the backward
pass allocate a full-size gradient per timestep. On a GPU every one of those small operations
is a kernel launch, so the step is bound by their number rather than by the arithmetic.

Arms, all over the same ``torchcrf.CRF`` parameters and all giving the same loss and gradients:

  pytorch-crf  what delft runs today
  A            score of the gold path without a loop; normalizer loop over ``unbind``
  B            A, with the normalizer loop scripted (what #230 did for the decode)
  D            scripted alpha and beta recursions with a hand-written backward pass
               (forward-backward algorithm), so no autograd graph is built in the loops

On ``--device cuda`` every arm is timed with the CRF on the GPU and with the CRF on the CPU
under a model on the GPU (emissions are moved, gradients flow back to the GPU).

It needs torch and pytorch-crf only: no corpus, no model, and no import of delft.

Usage:
  python scripts/benchmark_crf_loss.py                       # cuda when available, else cpu
  python scripts/benchmark_crf_loss.py --device cpu --threads 4
  python scripts/benchmark_crf_loss.py --shape citation:200,100,37,20 --reps 10
  python scripts/benchmark_crf_loss.py --profile-steps 5 --json crf_loss.json
"""

import argparse
import json
import socket
import sys
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
from torchcrf import CRF as TorchCRF

REFERENCE_ARM = "pytorch-crf"

# name -> (batch size, sequence length, number of tags, shortest sequence)
# citation and reference-segmenter are the shapes of the runs measured on Vertex AI:
# batch 200 over windows of 100 tokens, and batch 30 over sequences of 2000 lines.
DEFAULT_SHAPES = {
    "citation": (200, 100, 37, 20),
    "header-like": (20, 100, 34, 20),
    "reference-segmenter": (30, 2000, 5, 200),
    "segmentation-like": (10, 2000, 15, 200),
}

# (batch size, sequence length, number of tags, shortest sequence) of the correctness checks
CHECK_SHAPES = [(7, 23, 6, 1), (4, 50, 37, 5)]
CHECK_TOLERANCE = {torch.float64: 1e-9, torch.float32: 1e-4}

WARMUP_STEPS = 3


# --- arms --------------------------------------------------------------------------------
# Every arm takes batch-first emissions [B,T,K], tags [B,T] and a bool mask [B,T] whose
# first column is all on, and returns the mean negative log-likelihood, as delft's CRF does.


def loss_pytorch_crf(crf: TorchCRF, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return -crf(emissions, tags, mask=mask, reduction="mean")


def gold_path_score(crf: TorchCRF, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Score of the gold path, [B]. Nothing here depends on the previous timestep: no loop."""
    mask_values = mask.to(emissions.dtype)
    emitted = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)  # [B,T]
    moved = crf.transitions[tags[:, :-1], tags[:, 1:]]  # [B,T-1]
    score = crf.start_transitions[tags[:, 0]] + emitted[:, 0]
    score = score + ((moved + emitted[:, 1:]) * mask_values[:, 1:]).sum(dim=1)
    last_positions = mask.long().sum(dim=1) - 1
    last_tags = tags.gather(1, last_positions.unsqueeze(1)).squeeze(1)
    return score + crf.end_transitions[last_tags]


def normalizer_lean(crf: TorchCRF, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Log partition function, [B]. The recurrence stays, with fewer operations per timestep:
    the timesteps are unbound once (one gradient for the whole tensor instead of one per
    index), and the emission is added after the logsumexp, since it does not depend on the
    previous tag.
    """
    steps = emissions.transpose(0, 1).unbind(0)
    valid = mask.transpose(0, 1).unsqueeze(2).unbind(0)
    transitions = crf.transitions
    score = crf.start_transitions + steps[0]
    for i in range(1, len(steps)):
        next_score = torch.logsumexp(score.unsqueeze(2) + transitions, dim=1) + steps[i]
        score = torch.where(valid[i], next_score, score)
    return torch.logsumexp(score + crf.end_transitions, dim=1)


def loss_a(crf: TorchCRF, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return -(gold_path_score(crf, emissions, tags, mask) - normalizer_lean(crf, emissions, mask)).mean()


@torch.jit.script
def normalizer_scripted(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start_transitions: torch.Tensor,
    transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> torch.Tensor:
    steps = emissions.transpose(0, 1).unbind(0)
    valid = mask.transpose(0, 1).unsqueeze(2).unbind(0)
    score = start_transitions + steps[0]
    for i in range(1, len(steps)):
        next_score = torch.logsumexp(score.unsqueeze(2) + transitions, dim=1) + steps[i]
        score = torch.where(valid[i], next_score, score)
    return torch.logsumexp(score + end_transitions, dim=1)


def loss_b(crf: TorchCRF, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    normalizer = normalizer_scripted(emissions, mask, crf.start_transitions, crf.transitions, crf.end_transitions)
    return -(gold_path_score(crf, emissions, tags, mask) - normalizer).mean()


@torch.jit.script
def forward_scores(
    emissions: torch.Tensor, mask: torch.Tensor, start_transitions: torch.Tensor, transitions: torch.Tensor
) -> torch.Tensor:
    """alpha, [T,B,K] from time-first emissions; a padded position repeats the previous one."""
    steps = emissions.unbind(0)
    valid = mask.unsqueeze(2).unbind(0)
    score = start_transitions + steps[0]
    scores = [score]
    for i in range(1, len(steps)):
        next_score = torch.logsumexp(score.unsqueeze(2) + transitions, dim=1) + steps[i]
        score = torch.where(valid[i], next_score, score)
        scores.append(score)
    return torch.stack(scores)


@torch.jit.script
def backward_scores(
    emissions: torch.Tensor, mask: torch.Tensor, transitions: torch.Tensor, end_transitions: torch.Tensor
) -> torch.Tensor:
    """beta, [T,B,K]: at and after the last real position of a sequence it is the end score."""
    steps = emissions.unbind(0)
    valid = mask.unsqueeze(2).unbind(0)
    seq_length = len(steps)
    score = end_transitions.unsqueeze(0).expand(steps[0].shape[0], -1)
    scores = [score]
    for k in range(1, seq_length):
        i = seq_length - k  # the score at i - 1, from the one at i
        previous_score = torch.logsumexp(transitions + (steps[i] + score).unsqueeze(1), dim=2)
        score = torch.where(valid[i], previous_score, score)
        scores.append(score)
    scores.reverse()
    return torch.stack(scores)


class LogPartition(torch.autograd.Function):
    """
    Log partition function over time-first emissions, with its gradient written out.

    The gradient of the log partition function is the marginals: of the tag at each position
    for the emissions, and of each pair of consecutive tags for the transitions. They come
    from the forward and backward scores, so neither recursion needs an autograd graph.
    """

    @staticmethod
    def forward(ctx, emissions, mask, start_transitions, transitions, end_transitions):
        alphas = forward_scores(emissions, mask, start_transitions, transitions)
        log_partition = torch.logsumexp(alphas[-1] + end_transitions, dim=1)
        ctx.save_for_backward(emissions, mask, transitions, end_transitions, alphas, log_partition)
        return log_partition

    @staticmethod
    def backward(ctx, grad_output):
        emissions, mask, transitions, end_transitions, alphas, log_partition = ctx.saved_tensors
        betas = backward_scores(emissions, mask, transitions, end_transitions)
        mask_values = mask.to(emissions.dtype)
        per_sequence = grad_output.unsqueeze(0).unsqueeze(2)  # [1,B,1]
        log_z = log_partition.unsqueeze(0).unsqueeze(2)  # [1,B,1]

        tag_marginals = torch.exp(alphas + betas - log_z) * mask_values.unsqueeze(2)
        grad_emissions = tag_marginals * per_sequence
        grad_start = grad_emissions[0].sum(dim=0)
        grad_end = (
            torch.exp(alphas[-1] + end_transitions - log_partition.unsqueeze(1)) * grad_output.unsqueeze(1)
        ).sum(dim=0)

        # Pair marginals summed over positions and sequences, as one matrix product:
        # exp(alpha[t-1,i] + transitions[i,j] + emissions[t,j] + beta[t,j] - logZ), each side
        # shifted by its own maximum so that neither exponential leaves float range.
        left = alphas[:-1]
        right = emissions[1:] + betas[1:]
        left_max = left.max(dim=2, keepdim=True).values
        right_max = right.max(dim=2, keepdim=True).values
        weight = torch.exp(left_max + right_max - log_z) * mask_values[1:].unsqueeze(2) * per_sequence
        num_tags = emissions.shape[2]
        left_values = (torch.exp(left - left_max) * weight).reshape(-1, num_tags)
        right_values = torch.exp(right - right_max).reshape(-1, num_tags)
        grad_transitions = torch.exp(transitions) * (left_values.t() @ right_values)

        return grad_emissions, None, grad_start, grad_transitions, grad_end


def loss_d(crf: TorchCRF, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    normalizer = LogPartition.apply(
        emissions.transpose(0, 1), mask.transpose(0, 1), crf.start_transitions, crf.transitions, crf.end_transitions
    )
    return -(gold_path_score(crf, emissions, tags, mask) - normalizer).mean()


LossFunction = Callable[[TorchCRF, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

ARMS: Dict[str, Tuple[str, LossFunction]] = {
    REFERENCE_ARM: ("pytorch-crf (today)", loss_pytorch_crf),
    "A": ("A  score without a loop, lean loop", loss_a),
    "B": ("B  A, with the loop scripted", loss_b),
    "D": ("D  scripted loops, own backward", loss_d),
}


# --- data --------------------------------------------------------------------------------


class Batch(NamedTuple):
    crf: TorchCRF
    emissions: torch.Tensor
    tags: torch.Tensor
    mask: torch.Tensor


def make_batch(
    shape: Tuple[int, int, int, int],
    dtype: torch.dtype,
    model_device: torch.device,
    crf_device: torch.device,
    seed: int,
) -> Batch:
    """
    A batch as the model would hand it to the CRF: emissions (a leaf, standing for the output
    of the layers below), tags and mask on the device of the model, the CRF on its own.
    Sequences are prefixes of varying length, one of them full, as pytorch-crf requires.
    """
    batch_size, seq_length, num_tags, min_length = shape
    generator = torch.Generator().manual_seed(seed)
    crf = TorchCRF(num_tags, batch_first=True).to(dtype)
    with torch.no_grad():
        for parameter in crf.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=dtype))
    emissions = torch.randn(batch_size, seq_length, num_tags, generator=generator, dtype=dtype)
    tags = torch.randint(0, num_tags, (batch_size, seq_length), generator=generator)
    lengths = torch.randint(min_length, seq_length + 1, (batch_size,), generator=generator)
    lengths[0] = seq_length
    mask = torch.arange(seq_length).unsqueeze(0) < lengths.unsqueeze(1)
    return Batch(
        crf.to(crf_device),
        emissions.to(model_device).requires_grad_(True),
        tags.to(model_device),
        mask.to(model_device),
    )


def compute_loss(loss_function: LossFunction, batch: Batch) -> torch.Tensor:
    """The loss of a batch, its inputs moved to the device of the CRF when that is another one."""
    crf_device = batch.crf.transitions.device
    return loss_function(
        batch.crf, batch.emissions.to(crf_device), batch.tags.to(crf_device), batch.mask.to(crf_device)
    )


def zero_gradients(batch: Batch) -> None:
    batch.crf.zero_grad()
    batch.emissions.grad = None


def synchronize(*devices: torch.device) -> None:
    if any(device.type == "cuda" for device in devices):
        torch.cuda.synchronize()


# --- TorchScript on a GPU ----------------------------------------------------------------

# The loop of the scripted arms in miniature. Compiled anew for every probe, since a
# scripted function keeps the execution plan of its first runs.
FUSION_PROBE_SOURCE = """
def probe(steps: Tensor, valid: Tensor, transitions: Tensor) -> Tensor:
    score = steps[0]
    for i in range(1, steps.size(0)):
        next_score = torch.logsumexp(score.unsqueeze(2) + transitions, dim=1) + steps[i]
        score = torch.where(valid[i], next_score, score)
    return score
"""


def scripted_loop_error_on_gpu() -> Optional[str]:
    """The first line of the error a scripted loop fails with on the GPU, or None when it runs."""
    probe = torch.jit.CompilationUnit(FUSION_PROBE_SOURCE).probe
    valid = torch.ones(6, 4, 1, dtype=torch.bool, device="cuda")
    try:
        for dtype in (torch.float64, torch.float32):
            steps = torch.randn(6, 4, 5, dtype=dtype, device="cuda", requires_grad=True)
            transitions = torch.randn(5, 5, dtype=dtype, device="cuda", requires_grad=True)
            # the optimized plan, with its fused kernels, is built after the profiled runs
            for _ in range(4):
                probe(steps, valid, transitions).sum().backward()
        torch.cuda.synchronize()
    except RuntimeError as error:
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        return next((line for line in lines if "nvrtc" in line), lines[0] if lines else repr(error))
    return None


def configure_gpu_fusion(mode: str) -> str:
    """
    Whether TorchScript fuses the element-wise operations of a scripted loop into one GPU
    kernel. It compiles such a kernel with nvrtc when the loop first runs, and an
    installation without the nvrtc builtins fails right there. Under `auto` the fusion is
    turned off when it does not work, and the scripted arms then run their operations one
    by one: fewer interpreter steps than Python, the same number of kernel launches.
    """

    def turn_off() -> None:
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)

    if mode == "off":
        turn_off()
        return "off"
    error = scripted_loop_error_on_gpu()
    if error is None:
        return "on"
    if mode == "on":
        sys.exit(f"--jit-gpu-fusion on was asked for and a scripted loop does not run on this GPU: {error}")
    turn_off()
    remaining_error = scripted_loop_error_on_gpu()
    if remaining_error is not None:
        sys.exit(
            f"a scripted loop does not run on this GPU, with or without fusion: {remaining_error}\n"
            f"the arms without TorchScript still run: --arms {REFERENCE_ARM},A"
        )
    return f"off, the fused kernels do not compile here ({error})"


# --- correctness -------------------------------------------------------------------------


def loss_and_gradients(loss_function: LossFunction, batch: Batch) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    zero_gradients(batch)
    loss = compute_loss(loss_function, batch)
    loss.backward()
    gradients = [batch.emissions.grad] + [parameter.grad for parameter in batch.crf.parameters()]
    return loss.detach().cpu(), [gradient.detach().cpu().clone() for gradient in gradients]


def check_arms(arm_names: List[str], model_device: torch.device, crf_device: torch.device) -> None:
    """Every arm must give the loss and the gradients of pytorch-crf, or nothing is timed."""
    for dtype, tolerance in CHECK_TOLERANCE.items():
        for shape in CHECK_SHAPES:
            batch = make_batch(shape, dtype, model_device, crf_device, seed=0)
            expected_loss, expected_gradients = loss_and_gradients(loss_pytorch_crf, batch)
            for name in arm_names:
                if name == REFERENCE_ARM:
                    continue
                loss, gradients = loss_and_gradients(ARMS[name][1], batch)
                loss_difference = (loss - expected_loss).abs().item()
                gradient_difference = max(
                    (gradient - expected).abs().max().item()
                    for gradient, expected in zip(gradients, expected_gradients)
                )
                if not max(loss_difference, gradient_difference) <= tolerance:
                    sys.exit(
                        f"arm {name} differs from pytorch-crf on shape {shape}, {dtype}, CRF on {crf_device}: "
                        f"loss by {loss_difference:.1e}, gradients by {gradient_difference:.1e} "
                        f"(tolerance {tolerance:.0e})"
                    )
    print(f"check: {', '.join(n for n in arm_names if n != REFERENCE_ARM)} match pytorch-crf, CRF on {crf_device}")


# --- timing ------------------------------------------------------------------------------


def time_arm(loss_function: LossFunction, batch: Batch, reps: int) -> Tuple[float, float, Optional[float]]:
    """Mean milliseconds of the forward and of the backward pass, and peak GPU memory in MB."""
    model_device = batch.emissions.device
    crf_device = batch.crf.transitions.device
    for _ in range(WARMUP_STEPS):
        zero_gradients(batch)
        compute_loss(loss_function, batch).backward()
    if model_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    forward_seconds = backward_seconds = 0.0
    for _ in range(reps):
        zero_gradients(batch)
        synchronize(model_device, crf_device)
        start = time.perf_counter()
        loss = compute_loss(loss_function, batch)
        synchronize(model_device, crf_device)
        middle = time.perf_counter()
        loss.backward()
        synchronize(model_device, crf_device)
        end = time.perf_counter()
        forward_seconds += middle - start
        backward_seconds += end - middle
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if model_device.type == "cuda" else None
    return 1000 * forward_seconds / reps, 1000 * backward_seconds / reps, peak_mb


def profile_arm(loss_function: LossFunction, batch: Batch, steps: int, top: int = 12) -> None:
    """Calls and self time per operator over a few steps: the count is what a GPU pays for."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if batch.emissions.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    for _ in range(WARMUP_STEPS):
        zero_gradients(batch)
        compute_loss(loss_function, batch).backward()
    with profile(activities=activities) as profiler:
        for _ in range(steps):
            zero_gradients(batch)
            compute_loss(loss_function, batch).backward()
        synchronize(batch.emissions.device)
    events = profiler.key_averages()

    def device_time(event) -> float:
        return getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0))

    total_calls = sum(event.count for event in events)
    print(f"    {total_calls / steps:10.0f} operator calls per step")
    print(f"    {'operator':42s} {'calls/step':>10s} {'self cpu ms':>12s} {'self gpu ms':>12s}")
    for event in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:top]:
        print(
            f"    {event.key[:42]:42s} {event.count / steps:10.1f} "
            f"{event.self_cpu_time_total / 1000 / steps:12.2f} {device_time(event) / 1000 / steps:12.2f}"
        )


# --- command line ------------------------------------------------------------------------


def parse_shape(value: str) -> Tuple[str, Tuple[int, int, int, int]]:
    try:
        name, numbers = value.split(":")
        batch_size, seq_length, num_tags, min_length = (int(number) for number in numbers.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected NAME:B,T,K,MINLEN, got '{value}'") from None
    if not (batch_size >= 1 and seq_length >= 2 and num_tags >= 1 and 1 <= min_length <= seq_length):
        raise argparse.ArgumentTypeError(f"need B >= 1, T >= 2, K >= 1 and 1 <= MINLEN <= T, got '{value}'")
    return name, (batch_size, seq_length, num_tags, min_length)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        "--device", default=None, help="device of the model: cuda or cpu (default: cuda when available)"
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        metavar="NAME:B,T,K,MINLEN",
        help="batch size, sequence length, number of tags and shortest sequence; may be repeated "
        f"(default: {', '.join(DEFAULT_SHAPES)})",
    )
    parser.add_argument("--arms", default=",".join(ARMS), help=f"comma-separated arms (default: {','.join(ARMS)})")
    parser.add_argument("--reps", type=int, default=5, help="timed steps per arm, after 3 warm-up steps (default: 5)")
    parser.add_argument("--threads", type=int, default=None, help="intra-op CPU threads (default: torch's own)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile-steps", type=int, default=0, help="also profile this many steps of pytorch-crf and A"
    )
    parser.add_argument("--json", default=None, metavar="PATH", help="also write the records as JSON")
    parser.add_argument(
        "--jit-gpu-fusion",
        choices=["auto", "on", "off"],
        default="auto",
        help="let TorchScript fuse operations into GPU kernels; auto turns it off where they do not compile",
    )
    parser.add_argument("--skip-check", action="store_true", help="do not compare the arms with pytorch-crf first")
    args = parser.parse_args()
    args.arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in args.arms if name not in ARMS]
    if unknown:
        parser.error(f"unknown arms {unknown}; known: {list(ARMS)}")
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    return args


def describe_environment(model_device: torch.device) -> Dict[str, object]:
    environment: Dict[str, object] = {
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "device": str(model_device),
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
    }
    if model_device.type == "cuda":
        major, minor = torch.cuda.get_device_capability()
        environment.update(
            gpu=torch.cuda.get_device_name(),
            gpu_capability=f"sm_{major}{minor}",
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
        )
    return environment


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model_device = torch.device(args.device)
    if model_device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda was asked for and torch sees no GPU")

    environment = describe_environment(model_device)
    if model_device.type == "cuda":
        environment["jit_gpu_fusion"] = configure_gpu_fusion(args.jit_gpu_fusion)
    print(" | ".join(f"{key}={value}" for key, value in environment.items()))

    # where the CRF runs under a model on model_device
    if model_device.type == "cuda":
        placements = [("CRF on the GPU", model_device), ("CRF on the CPU, model on the GPU", torch.device("cpu"))]
    else:
        placements = [("CRF on the CPU", model_device)]

    if not args.skip_check:
        for _, crf_device in placements:
            check_arms(args.arms, model_device, crf_device)

    shapes = dict(args.shape) if args.shape else DEFAULT_SHAPES
    records = []
    for shape_name, shape in shapes.items():
        batch_size, seq_length, num_tags, _ = shape
        print(f"\n=== {shape_name}: batch {batch_size}, {seq_length} positions, {num_tags} tags")
        for placement_name, crf_device in placements:
            print(f"--- {placement_name}  (ms per step: forward, backward, total; speed-up; peak GPU MB)")
            batch = make_batch(shape, torch.float32, model_device, crf_device, seed=args.seed)
            reference_total = None
            for name in args.arms:
                label, loss_function = ARMS[name]
                forward_ms, backward_ms, peak_mb = time_arm(loss_function, batch, args.reps)
                total_ms = forward_ms + backward_ms
                if name == REFERENCE_ARM:
                    reference_total = total_ms
                speedup = reference_total / total_ms if reference_total else None
                print(
                    f"{label:36s} {forward_ms:9.1f} {backward_ms:9.1f} {total_ms:9.1f}"
                    f"   {f'{speedup:5.2f}x' if speedup else '     -':>6s}"
                    f"   {f'{peak_mb:8.0f}' if peak_mb is not None else '':>8s}"
                )
                records.append(
                    {
                        "shape": shape_name,
                        "batch_size": batch_size,
                        "seq_length": seq_length,
                        "num_tags": num_tags,
                        "placement": placement_name,
                        "crf_device": str(crf_device),
                        "arm": name,
                        "forward_ms": forward_ms,
                        "backward_ms": backward_ms,
                        "total_ms": total_ms,
                        "speedup": speedup,
                        "peak_gpu_mb": peak_mb,
                    }
                )
            if args.profile_steps > 0:
                for name in (REFERENCE_ARM, "A"):
                    if name in args.arms:
                        print(f"  profile of {ARMS[name][0].strip()}, {placement_name}:")
                        profile_arm(ARMS[name][1], batch, args.profile_steps)

    if args.json:
        with open(args.json, "w") as output:
            json.dump({"environment": environment, "reps": args.reps, "records": records}, output, indent=2)
        print(f"\nrecords written to {args.json}")


if __name__ == "__main__":
    main()
