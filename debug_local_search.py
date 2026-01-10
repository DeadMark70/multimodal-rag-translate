import importlib.util
import sys
import os

file_path = os.path.join("graph_rag", "local_search.py")
module_name = "graph_rag.local_search"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
# We need to mock the imports that fail inside the module
sys.modules["graph_rag.store"] = type("Mock", (object,), {"GraphStore": object})
sys.modules["core.llm_factory"] = type("Mock", (object,), {"get_llm": lambda x: None})
sys.modules["graph_rag.schemas"] = type("Mock", (object,), {"EntityType": object, "GraphNode": object})
sys.modules["langchain_core.messages"] = type("Mock", (object,), {"HumanMessage": object})

try:
    spec.loader.exec_module(module)
    print("Module loaded successfully.")
except Exception as e:
    print(f"Module load error: {e}")

print("Attributes of local_search module:")
print(dir(module))

if hasattr(module, "local_search"):
    print("\nlocal_search function found.")
else:
    print("\nlocal_search function NOT found.")

