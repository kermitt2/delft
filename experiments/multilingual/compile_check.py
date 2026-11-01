import os, sys
from delft import DELFT_PROJECT_DIR
from delft.utilities.Embeddings import Embeddings, load_resource_registry
r = load_resource_registry(os.path.join(DELFT_PROJECT_DIR, "resources-registry.json"))
print(DELFT_PROJECT_DIR)
for name in sys.argv[1:]:
    e = Embeddings(name, resource_registry=r)
    v = lambda w: bool(abs(e.get_word_vector(w)).sum() > 0)
    print(name, e.embed_size, {w: v(w) for w in ["the", "University", "고찰", "Universität", "zzqxv"]}, flush=True)
