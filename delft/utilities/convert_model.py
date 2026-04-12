"""
Convert pre-TF-2.17 DeLFT models to the current TF 2.17 / ``tf_keras`` 2.17 stack.

DeLFT only persists weights (``model.save_weights()``), not the full Keras
architecture. The architecture is rebuilt from ``config.json`` (and
``preprocessor.json`` for sequence labelling). That makes cross-version
conversion a weight-remapping problem rather than a deserialisation problem.

The converter:

1. Copies JSON / tokenizer artifacts from the source directory to the output
   directory unchanged.
2. Rewrites ``model_name`` in the copied ``config.json`` to match the output
   directory's basename, so the standard DeLFT loaders can pick it up.
3. Builds a fresh Keras model from that copied config using the current
   ``tf_keras`` code paths.
4. Opens the old ``model_weights.hdf5`` with ``h5py`` and assigns each tensor
   into the corresponding variable of the fresh model, applying
   ``RENAME_RULES`` when direct name lookup fails.
5. Saves the new weights file alongside the copied artifacts.

Usage::

    python -m delft.utilities.convert_model \\
        --input data/models/sequenceLabelling/grobid-quantities-BERT_CRF \\
        --output data/models/sequenceLabelling/grobid-quantities-BERT_CRF-tf217 \\
        [--task auto|seq|class] [--dry-run] [--verify]

The converter does not guarantee *numerical* equivalence — only that the new
model loads, builds, and produces finite output on a dummy forward pass. Users
should re-run their own evaluation pipeline on the converted model before
relying on it in production.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

DEFAULT_WEIGHT_FILE = "model_weights.hdf5"
CONFIG_FILE = "config.json"
PROCESSOR_FILE = "preprocessor.json"
TRANSFORMER_CONFIG_FILE = "transformer-config.json"
TRANSFORMER_TOKENIZER_DIR = "transformer-tokenizer"


# =============================================================================
# Rename rules (reviewable — extend as real mismatches surface)
# =============================================================================
#
# Each rule is an ``(old_pattern, new_replacement)`` pair applied via
# ``re.sub`` to a *fresh-model variable name*. The converter tries the rule
# output against the flat index of the old HDF5 file; on a hit it copies the
# weight.
#
# Two generic fallbacks are already built into :func:`find_old_key`:
#   * add / strip the trailing ``:0`` tensor-ordinal suffix
#   * strip numeric ``_<n>`` suffixes from layer names (``dense_1`` -> ``dense``)
#
# Those cover the bulk of cross-version drift we observed between the Keras
# 2.9.0 HDF5 in ``data/models/sequenceLabelling/grobid-quantities-BERT_CRF``
# and the current Keras 2.17.0 files (top-level layout, CRF internals and
# LSTM cell naming are all stable). Keep this table *short*: every rule is a
# potential false positive, so only add entries that fix a confirmed failure.
#
# To add a rule, append a ``(pattern, replacement)`` tuple. Anchor the
# pattern so it only matches the intended substring, e.g.::
#
#     (r"^model/tf_bert_model/", "model/tf_roberta_model/"),
#
RENAME_RULES: list[tuple[str, str]] = [
    # Empty by default. The two generic fallbacks in find_old_key() handle
    # the drift observed in existing pre-upgrade models.
]


# =============================================================================
# Unmatched-weight policy (abort by default; --force-partial opts in)
# =============================================================================
# Module-level flag set from the CLI. When False (the default), any fresh
# variable with no counterpart in the old HDF5 aborts the conversion. When
# True (--force-partial), we warn and leave the variable at its Keras init
# value — useful if the new architecture legitimately added a layer.
FORCE_PARTIAL: bool = False


def _handle_unmatched(var_name: str, var_shape: tuple, close_matches: list[str]) -> str:
    hint = ""
    if close_matches:
        hint = "\n  Close matches in old HDF5:\n" + "\n".join(f"    - {k}" for k in close_matches)
    if FORCE_PARTIAL:
        print(f"    WARN unmatched: {var_name!r} shape={var_shape} → keeping Keras init{hint}")
        return "keep_init"
    raise ValueError(
        f"No old-HDF5 match for fresh variable {var_name!r} shape={var_shape}.\n"
        f"{hint}\n"
        "Extend RENAME_RULES, or pass --force-partial to leave it at its Keras "
        "init value."
    )


# =============================================================================
# Task autodetection
# =============================================================================
def detect_task(input_dir: Path) -> str:
    parent = input_dir.parent.name
    if parent == "sequenceLabelling":
        return "seq"
    if parent == "textClassification":
        return "class"
    # Fallback: sequence labelling models always ship a preprocessor.json,
    # text classification models never do.
    if (input_dir / PROCESSOR_FILE).exists():
        print(f"  auto-detect: found {PROCESSOR_FILE} → assuming sequence labelling")
        return "seq"
    if (input_dir / CONFIG_FILE).exists():
        print(f"  auto-detect: no {PROCESSOR_FILE} → assuming text classification")
        return "class"
    raise ValueError(
        f"Could not auto-detect task from parent directory {parent!r} "
        f"or directory contents. Pass --task seq or --task class explicitly."
    )


# =============================================================================
# Artifact copy
# =============================================================================

# Keys in tokenizer_config.json that conflict with methods added in newer
# versions of the ``transformers`` library. Removing them is safe — they are
# ``from_pretrained()`` parameters, not tokenizer config.
CONFLICTING_TOKENIZER_KEYS = ["add_special_tokens"]


def copy_artifacts(src: Path, dst: Path, task: str, redownload_tokenizer: bool = False) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    required = [CONFIG_FILE]
    if task == "seq":
        required.append(PROCESSOR_FILE)

    for filename in required:
        src_f = src / filename
        if not src_f.exists():
            raise FileNotFoundError(f"Required file missing in source: {src_f}")
        shutil.copy2(src_f, dst / filename)
        print(f"  copied {filename}")

    for optional in (TRANSFORMER_CONFIG_FILE,):
        src_f = src / optional
        if src_f.exists():
            shutil.copy2(src_f, dst / optional)
            print(f"  copied {optional}")

    if redownload_tokenizer:
        print(f"  skipped {TRANSFORMER_TOKENIZER_DIR}/ (will re-download from Hub)")
    else:
        tok_src = src / TRANSFORMER_TOKENIZER_DIR
        if tok_src.is_dir():
            tok_dst = dst / TRANSFORMER_TOKENIZER_DIR
            if tok_dst.exists():
                shutil.rmtree(tok_dst)
            shutil.copytree(tok_src, tok_dst)
            print(f"  copied {TRANSFORMER_TOKENIZER_DIR}/")


def rewrite_model_name(dst: Path) -> str:
    """Update config.json's ``model_name`` to match the output directory basename."""
    config_path = dst / CONFIG_FILE
    with open(config_path) as f:
        cfg = json.load(f)
    new_name = dst.name
    old_name = cfg.get("model_name")
    if old_name != new_name:
        cfg["model_name"] = new_name
        with open(config_path, "w") as f:
            json.dump(cfg, f, sort_keys=False, indent=4)
        print(f"  rewrote config.json model_name: {old_name!r} -> {new_name!r}")
    return new_name


def patch_tokenizer_config(dst: Path) -> None:
    """Remove keys from ``tokenizer_config.json`` that conflict with newer ``transformers``."""
    tc_path = dst / TRANSFORMER_TOKENIZER_DIR / "tokenizer_config.json"
    if not tc_path.exists():
        return
    with open(tc_path) as f:
        cfg = json.load(f)
    removed = [k for k in CONFLICTING_TOKENIZER_KEYS if k in cfg]
    if not removed:
        return
    for key in removed:
        del cfg[key]
    with open(tc_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  patched tokenizer_config.json: removed conflicting keys {removed}")


def download_fresh_tokenizer(config_path: Path, dst: Path) -> None:
    """Download a fresh tokenizer from HuggingFace Hub and save to the output dir."""
    with open(config_path) as f:
        cfg = json.load(f)
    name = cfg.get("transformer_name")
    if not name:
        raise ValueError("config.json has no transformer_name — cannot download tokenizer")
    from transformers import AutoTokenizer

    print(f"  downloading fresh tokenizer for {name!r} from HuggingFace Hub")
    tok = AutoTokenizer.from_pretrained(name)
    tok_dir = dst / TRANSFORMER_TOKENIZER_DIR
    tok.save_pretrained(str(tok_dir))
    print(f"  saved to {tok_dir}")


# =============================================================================
# Source HDF5 sanity check
# =============================================================================
def check_source_hdf5(path: Path) -> dict:
    import h5py

    with h5py.File(str(path), "r") as h:
        raw_v = h.attrs.get("keras_version")
        if isinstance(raw_v, bytes):
            raw_v = raw_v.decode()
        raw_b = h.attrs.get("backend")
        if isinstance(raw_b, bytes):
            raw_b = raw_b.decode()
        top = list(h.keys())

    if raw_v is None:
        raise ValueError(
            f"Source weights file has no 'keras_version' attribute: {path}. "
            "This is not a recognisable Keras weights file."
        )
    major = int(str(raw_v).split(".")[0])
    if major < 2:
        raise ValueError(
            f"Pre-Keras-2.x weights format is not supported (found keras_version={raw_v}). "
            "Retrain the model from scratch."
        )
    return {"keras_version": raw_v, "backend": raw_b, "top_keys": top}


# =============================================================================
# HDF5 flatten + name lookup
# =============================================================================
def flatten_h5_weights(h5_root) -> dict:
    """Walk an HDF5 file and return a flat ``{full_name: numpy_array}`` dict.

    Names use ``/`` separators, e.g.
    ``model/bidirectional_1/forward_lstm_1/lstm_cell/kernel:0``.
    """
    out: dict = {}

    def walk(g, prefix: str = ""):
        for key in g.keys():
            child = g[key]
            full = f"{prefix}{key}"
            if hasattr(child, "shape"):
                out[full] = child[()]
            else:
                walk(child, full + "/")

    walk(h5_root)
    return out


def apply_rename_rules(name: str, rules: list[tuple[str, str]]) -> str:
    for pattern, replacement in rules:
        name = re.sub(pattern, replacement, name)
    return name


def build_suffix_index(old_index: dict) -> dict[str, list[str]]:
    """Build a reverse lookup: for every old key, register all its suffixes.

    For ``crf/chain_kernel:0`` this registers ``crf/chain_kernel:0`` and
    ``chain_kernel:0``.  For ``model/tf_bert_model/bert/.../weight:0`` it
    registers every sub-path from the full path down to ``weight:0``.

    Returns ``{suffix: [list of full old keys that end with it]}``.
    """
    idx: dict[str, list[str]] = {}
    for full_key in old_index:
        parts = full_key.split("/")
        for i in range(len(parts)):
            suffix = "/".join(parts[i:])
            idx.setdefault(suffix, []).append(full_key)
    return idx


def find_old_key(
    new_name: str,
    old_index: dict,
    suffix_index: dict[str, list[str]],
    top_groups: list[str],
    target_shape: Optional[tuple] = None,
) -> Optional[str]:
    """Try progressively looser strategies to match a fresh variable name to an old HDF5 key.

    Strategy order:
      1. Exact match on full path.
      2. Apply RENAME_RULES then exact match.
      3. Toggle ``:0`` suffix then exact match.
      4. Drop numeric ``_<n>`` suffixes from layer names then exact match.
      5. Prepend each top-level HDF5 group name (handles the Keras-2 pattern
         where save_weights creates ``<layer>/<layer>/var`` paths while
         Keras var names are ``<layer>/var``).
      6. Suffix match (unique hit only).
      7. Suffix match with shape disambiguation (when multiple hits).
    """
    # ── direct-hit strategies ──
    candidates = [
        new_name,
        apply_rename_rules(new_name, RENAME_RULES),
        new_name.removesuffix(":0") if new_name.endswith(":0") else new_name + ":0",
        re.sub(r"(_\d+)(?=/|:)", "", new_name),
    ]
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c in old_index:
            return c

    # ── prepend top-level HDF5 group strategy ──
    # Keras-2 save_weights writes <group>/<var_name> where group is the layer
    # name and var_name already starts with the layer name, producing paths
    # like dense/dense/kernel:0.  The fresh var.name is just dense/kernel:0.
    for c in candidates:
        for grp in top_groups:
            prefixed = f"{grp}/{c}"
            if prefixed in old_index:
                return prefixed

    # ── numeric-suffix remapping strategy ──
    # When the old model was the Nth instance in a session, Keras auto-numbers
    # all layer names (e.g. tf_bert_model_1, dense_1). The fresh model is the
    # 1st instance so has no suffix. Map fresh base names to old suffixed names.
    for c in candidates:
        first_seg = c.split("/")[0] if "/" in c else c.split(":")[0]
        first_seg_base = re.sub(r"_\d+$", "", first_seg)
        for grp in top_groups:
            grp_base = re.sub(r"_\d+$", "", grp)
            if grp_base == first_seg_base and grp != first_seg:
                # Replace first segment with the suffixed group name, then
                # prepend as the top-level group (doubled-prefix pattern).
                rest = c[len(first_seg) :]
                mapped = grp + rest  # e.g. tf_bert_model_1/bert/.../weight:0
                if mapped in old_index:
                    return mapped
                doubled = f"{grp}/{mapped}"  # e.g. tf_bert_model_1/tf_bert_model_1/bert/...
                if doubled in old_index:
                    return doubled

    # ── suffix-match strategy ──
    for c in candidates:
        matches = suffix_index.get(c, [])
        if len(matches) == 1:
            return matches[0]

    # ── suffix match with shape disambiguation ──
    if target_shape is not None:
        for c in candidates:
            matches = suffix_index.get(c, [])
            if len(matches) > 1:
                by_shape = [k for k in matches if tuple(old_index[k].shape) == target_shape]
                if len(by_shape) == 1:
                    return by_shape[0]

    return None


# =============================================================================
# Core: remap weights from old HDF5 into a fresh Keras model
# =============================================================================
def _find_close_matches(name: str, old_keys: list[str], max_results: int = 5) -> list[str]:
    """Return old keys whose leaf segment matches the leaf of *name*."""
    leaf = name.rsplit("/", 1)[-1]
    return [k for k in old_keys if k.endswith("/" + leaf) or k == leaf][:max_results]


def remap_weights(fresh_keras_model, old_hdf5_path: Path) -> dict:
    import h5py
    import numpy as np

    summary: dict[str, list] = {
        "matched": [],
        "unmatched": [],
        "shape_mismatches": [],
    }

    print("  step 1: native by_name tolerant load")
    try:
        fresh_keras_model.load_weights(str(old_hdf5_path), by_name=True, skip_mismatch=True)
        print("    native load completed")
    except Exception as e:  # noqa: BLE001
        print(f"    native load raised {type(e).__name__}: {e}")

    print("  step 2: h5py variable-level remap")
    with h5py.File(str(old_hdf5_path), "r") as h:
        old_index = flatten_h5_weights(h)
        top_groups = list(h.keys())

    suffix_index = build_suffix_index(old_index)

    for var in fresh_keras_model.weights:
        vname = var.name
        target_shape = tuple(var.shape.as_list()) if hasattr(var.shape, "as_list") else tuple(var.shape)

        old_key = find_old_key(vname, old_index, suffix_index, top_groups, target_shape)
        if old_key is None:
            close = _find_close_matches(vname, list(old_index.keys()))
            policy = _handle_unmatched(vname, target_shape, close)
            summary["unmatched"].append((vname, target_shape, close))
            if policy == "zero":
                var.assign(np.zeros(target_shape, dtype=var.dtype.as_numpy_dtype))
            continue

        old_arr = old_index[old_key]
        if tuple(old_arr.shape) != target_shape:
            summary["shape_mismatches"].append((vname, target_shape, tuple(old_arr.shape), old_key))
            continue

        var.assign(old_arr.astype(var.dtype.as_numpy_dtype))
        summary["matched"].append((vname, old_key))

    return summary


# =============================================================================
# Fresh wrapper construction
# =============================================================================
def build_fresh_seq(model_dir: Path):
    """Build a fresh Sequence wrapper against ``model_dir`` without loading weights."""
    from delft.sequenceLabelling.config import ModelConfig
    from delft.sequenceLabelling.models import get_model
    from delft.sequenceLabelling.preprocess import Preprocessor
    from delft.sequenceLabelling.wrapper import Sequence

    cfg = ModelConfig.load(str(model_dir / CONFIG_FILE))
    seq = Sequence(model_name=cfg.model_name)
    seq.model_config = cfg

    if cfg.embeddings_name is not None:
        seq.embeddings = seq.get_embedding(cfg.embeddings_name, use_cache=True)
        seq.model_config.word_embedding_size = seq.embeddings.embed_size
    else:
        seq.embeddings = None
        seq.model_config.word_embedding_size = 0

    seq.p = Preprocessor.load(str(model_dir / PROCESSOR_FILE))
    seq.model = get_model(
        seq.model_config,
        seq.p,
        ntags=len(seq.p.vocab_tag),
        load_pretrained_weights=False,
        local_path=str(model_dir),
        registry=seq.registry,
    )
    return seq


def build_fresh_class(model_dir: Path):
    from delft.textClassification.config import ModelConfig
    from delft.textClassification.models import getModel
    from delft.textClassification.wrapper import Classifier

    cfg = ModelConfig.load(str(model_dir / CONFIG_FILE))
    clf = Classifier(model_name=cfg.model_name)
    clf.model_config = cfg

    if getattr(cfg, "transformer_name", None) is None:
        clf.embeddings = clf.get_embedding(cfg.embeddings_name, use_cache=True)
        clf.model_config.word_embedding_size = clf.embeddings.embed_size
    else:
        clf.transformer_name = cfg.transformer_name
        clf.embeddings = None

    clf.model = getModel(
        clf.model_config,
        clf.training_config,
        load_pretrained_weights=False,
        local_path=str(model_dir),
        registry=clf.registry,
    )
    return clf


# =============================================================================
# Diagnostic helpers
# =============================================================================
def _report_build_failure(exc: Exception, model_dir: Path) -> None:
    """Print a diagnostic message when the fresh model build fails."""
    import traceback

    print("\n  BUILD FAILED — diagnostic info:")
    print(f"    Exception: {type(exc).__name__}: {exc}")

    try:
        import transformers as _tf

        print(f"    Installed transformers version: {_tf.__version__}")
    except ImportError:
        print("    transformers: not installed")
    try:
        import tf_keras as _tk

        print(f"    Installed tf_keras version:     {_tk.__version__}")
    except ImportError:
        print("    tf_keras: not installed")

    tc = model_dir / TRANSFORMER_CONFIG_FILE
    if tc.exists():
        import json as _json

        with open(tc) as f:
            tc_data = _json.load(f)
        saved_ver = tc_data.get("transformers_version", "(not recorded)")
        print(f"    Saved transformers_version:     {saved_ver}")

    if "add_special_tokens" in str(exc) or "tokenizer" in str(exc).lower():
        print(
            "\n  This looks like a HuggingFace tokenizer compatibility issue.\n"
            "  The saved transformer-tokenizer/ was created with a different\n"
            "  version of the `transformers` library than what is currently\n"
            "  installed.\n"
            "\n  Fix: re-run the converter with --redownload-tokenizer to\n"
            "  download a fresh tokenizer from HuggingFace Hub.\n"
        )
    else:
        print("\n  Full traceback:")
        traceback.print_exc()


# =============================================================================
# Forward-pass verification
# =============================================================================
def verify_forward_pass(keras_model, task: str) -> None:
    """Run the freshly-weighted model on a dummy batch.

    Only asserts executability + finite output. Does not check numerical
    correctness (the user must re-evaluate with real data).
    """
    import numpy as np

    try:
        # CRF-wrapped models are subclassed, so keras_model.inputs may be None.
        # Fall back to base_model.inputs if available.
        inputs_spec = keras_model.inputs
        if inputs_spec is None and hasattr(keras_model, "base_model"):
            inputs_spec = keras_model.base_model.inputs
        if inputs_spec is None:
            print("  verification skipped: model has no .inputs spec (subclassed model)")
            return

        dummy_inputs = []
        for inp in inputs_spec:
            shape = [1 if d is None else d for d in inp.shape]
            dtype = inp.dtype.as_numpy_dtype if hasattr(inp.dtype, "as_numpy_dtype") else "int32"
            dummy = np.zeros(shape, dtype=dtype)
            dummy_inputs.append(dummy)
        out = keras_model(dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0], training=False)
        if isinstance(out, (list, tuple)):
            tensors = list(out)
        else:
            tensors = [out]
        for t in tensors:
            arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
            if not np.isfinite(arr).all():
                raise RuntimeError("Forward pass produced non-finite values")
        print("  verification passed (forward pass executed, output finite)")
    except Exception as e:  # noqa: BLE001
        print(f"  verification FAILED: {type(e).__name__}: {e}")
        raise


# =============================================================================
# Main conversion driver
# =============================================================================
def convert(
    input_dir: Path,
    output_dir: Path,
    task: str,
    verify: bool = False,
    dry_run: bool = False,
    force_partial: bool = False,
    redownload_tokenizer: bool = False,
) -> None:
    global FORCE_PARTIAL
    FORCE_PARTIAL = force_partial

    if task == "auto":
        task = detect_task(input_dir)
        print(f"auto-detected task: {task}")

    src_hdf5 = input_dir / DEFAULT_WEIGHT_FILE
    if not src_hdf5.exists():
        raise FileNotFoundError(f"Missing source weights: {src_hdf5}")

    print(f"source: {input_dir}")
    print(f"target: {output_dir}")
    info = check_source_hdf5(src_hdf5)
    print(f"  source keras_version: {info['keras_version']}")
    print(f"  source backend:       {info['backend']}")
    print(f"  source top-level:     {info['top_keys']}")

    if dry_run:
        print("\ndry-run: no files written")
        return

    print("\n[1/4] copying artifacts")
    copy_artifacts(input_dir, output_dir, task, redownload_tokenizer=redownload_tokenizer)
    rewrite_model_name(output_dir)
    if redownload_tokenizer:
        download_fresh_tokenizer(output_dir / CONFIG_FILE, output_dir)
    else:
        patch_tokenizer_config(output_dir)

    print("\n[2/4] building fresh model architecture")
    try:
        if task == "seq":
            wrapper = build_fresh_seq(output_dir)
        else:
            wrapper = build_fresh_class(output_dir)
    except Exception as e:
        _report_build_failure(e, output_dir)
        raise
    keras_model = wrapper.model.model  # inner tf_keras.Model

    print("\n[3/4] remapping weights")
    summary = remap_weights(keras_model, src_hdf5)
    print(f"  matched:          {len(summary['matched'])}")
    print(f"  unmatched:        {len(summary['unmatched'])}")
    print(f"  shape_mismatches: {len(summary['shape_mismatches'])}")
    for name, new_shape, old_shape, old_key in summary["shape_mismatches"][:10]:
        print(f"    SHAPE MISMATCH: {name}  new={new_shape}  old={old_shape}  (old key: {old_key})")
    for name, shape, close in summary["unmatched"][:10]:
        close_str = ", ".join(close) if close else "(none)"
        print(f"    UNMATCHED: {name}  shape={shape}  close_matches=[{close_str}]")
    if summary["shape_mismatches"]:
        raise RuntimeError(f"{len(summary['shape_mismatches'])} shape mismatches prevent conversion")

    print("\n[4/4] saving converted weights")
    out_weights = output_dir / DEFAULT_WEIGHT_FILE
    keras_model.save_weights(str(out_weights))
    print(f"  wrote {out_weights}")

    if verify:
        print("\nverifying forward pass")
        verify_forward_pass(keras_model, task)

    print("\nConversion complete.")
    print("Load the converted model with its standard DeLFT loader, pointing at:")
    print(f"  dir_path={output_dir.parent}  model_name={output_dir.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pre-TF-2.17 DeLFT models to TF 2.17 / tf_keras 2.17")
    parser.add_argument("--input", required=True, help="Source model directory")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination model directory (must differ from --input)",
    )
    parser.add_argument(
        "--task",
        choices=["auto", "seq", "class"],
        default="auto",
        help="Task framework: seq(uenceLabelling), (text)class(ification), or auto-detect",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the source and report without writing any files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a forward pass on dummy input after conversion",
    )
    parser.add_argument(
        "--force-partial",
        action="store_true",
        help="Continue when a fresh-model variable has no counterpart in the old "
        "HDF5; leave such variables at their Keras init value instead of aborting",
    )
    parser.add_argument(
        "--redownload-tokenizer",
        action="store_true",
        help="Download a fresh tokenizer from HuggingFace Hub instead of copying "
        "the saved one. Use when the saved tokenizer is incompatible with the "
        "installed transformers version.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    if input_dir == output_dir:
        parser.error("--input and --output must differ (in-place conversion is not supported)")
    if not input_dir.is_dir():
        parser.error(f"--input is not a directory: {input_dir}")

    try:
        convert(
            input_dir,
            output_dir,
            args.task,
            verify=args.verify,
            dry_run=args.dry_run,
            force_partial=args.force_partial,
            redownload_tokenizer=args.redownload_tokenizer,
        )
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
