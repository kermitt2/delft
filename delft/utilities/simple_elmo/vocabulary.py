#! /bin/env python3

# Simple tool to extract vocabulary from the corpus
# for training ELMo models

import sys
from collections import Counter

THRESHOLD = int(sys.argv[1])  # How many top frequent words you want to keep?

words = Counter()

for line in sys.stdin:
    tokenized = line.strip().split()
    words.update(tokenized)

print("\n".join(["<S>", "</S>", "<UNK>"]))

print(f"Vocabulary size before pruning: {len(words)}", file=sys.stderr)

a = words.most_common(THRESHOLD)
for w in a:
    print(w[0])
