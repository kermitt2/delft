import os
import sys

from delft import DELFT_PROJECT_DIR
from delft.utilities.Embeddings import Embeddings, load_resource_registry

r = load_resource_registry(os.path.join(DELFT_PROJECT_DIR, "resources-registry.json"))
print(DELFT_PROJECT_DIR)


def nonzero(e, w):
    return bool(abs(e.get_word_vector(w)).sum() > 0)


for name in sys.argv[1:]:
    e = Embeddings(name, resource_registry=r)
    print(
        name,
        e.embed_size,
        {w: nonzero(e, w) for w in ["the", "University", "고찰", "Universität", "zzqxv"]},
        flush=True,
    )
