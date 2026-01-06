
import sys
from unittest.mock import MagicMock

# Mock known missing modules
sys.modules["postgrest"] = MagicMock()
sys.modules["postgrest.exceptions"] = MagicMock()

# Also mock supabase potentially if it's missing (though requirements said it's there)
# sys.modules["supabase"] = MagicMock() 

print("Attempting to import main...")
try:
    import main
    print("Import successful!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    import traceback
    traceback.print_exc()
