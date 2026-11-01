#!/usr/bin/env python3
"""Package BPEmb multi as a static embedding directory (tokenizer.json + model.safetensors),
then check it word by word against the reference pooling of the original SentencePiece model.

argv: <word2vec .bin> <sentencepiece .model> <output dir>
"""
import inspect, os, sys
from pathlib import Path

import numpy as np
import sentencepiece
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers.convert_slow_tokenizer import generate_merges

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_bpemb import read_w2v_bin

bin_path, model_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
sp = sentencepiece.SentencePieceProcessor(); sp.load(model_path)
n = sp.get_piece_size()
print("pieces", n, "type", [p for p in map(sp.id_to_piece, range(6))])
words, matrix = read_w2v_bin(bin_path)
aligned = np.zeros((n, matrix.shape[1]), dtype=np.float32)
missing = 0
for i in range(n):
    r = words.get(sp.id_to_piece(i))
    if r is None:
        missing += 1
    else:
        aligned[i] = matrix[r]
print("pieces without vector", missing)
vocab = {sp.id_to_piece(i): i for i in range(n)}
scores = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(n)]
print("generate_merges", inspect.signature(generate_merges))
sig = list(inspect.signature(generate_merges).parameters)
merges, _ = (lambda r: r if isinstance(r, tuple) and len(r) == 2 else (r, None))(
    generate_merges(vocab, scores) if len(sig) >= 2 else generate_merges(vocab)
)
print("merges", len(merges))
tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=sp.id_to_piece(sp.unk_id()), fuse_unk=True))
tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
tok.save(f"{out}/tokenizer.json")
save_file({"embeddings": aligned}, f"{out}/model.safetensors")
(Path(out) / "config.json").write_text('{"normalize": false}\n')

tokens = [l.split("\t")[0] for l in open(HERE / "data/vocab.txt", encoding="utf-8", errors="replace")]
same = diff = 0; cos = []; examples = []
for t in tokens:
    a = [sp.id_to_piece(i) for i in sp.encode_as_ids(t.lower())]
    b = tok.encode(t, add_special_tokens=False).tokens
    if a == b:
        same += 1
    else:
        diff += 1
        va = aligned[[vocab[p] for p in a]].mean(0); vb = aligned[[vocab[p] for p in b]].mean(0)
        c = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)); cos.append(c)
        if len(examples) < 12:
            examples.append((t, a[:6], b[:6]))
print("identical split", same, "different", diff, "mean cosine of the different ones", np.mean(cos) if cos else None)
for e in examples: print(e)
