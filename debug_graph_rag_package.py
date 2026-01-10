import sys
from unittest.mock import MagicMock

sys.modules["networkx"] = MagicMock()
sys.modules["leidenalg"] = MagicMock()
sys.modules["igraph"] = MagicMock()

import graph_rag
from graph_rag import local_search

print(f"graph_rag file: {graph_rag.__file__}")
print(f"local_search file: {local_search.__file__}")

print(f"Has local_search function: {hasattr(local_search, 'local_search')}")
print(f"Dir local_search: {dir(local_search)}")
