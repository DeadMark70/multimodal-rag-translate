import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from supabase_client import supabase

def verify_docs(user_id: str):
    if not supabase:
        print("Supabase client not initialized.")
        return

    print(f"Checking documents for user: {user_id}")
    
    # Query documents for the user
    try:
        response = supabase.table("documents") \
            .select("id, file_name, created_at") \
            .eq("user_id", user_id) \
            .execute()
        
        docs = response.data
        if not docs:
            print("No documents found for this user.")
            return

        print(f"Found {len(docs)} documents.")
        
        # Check for target docs
        targets = ["SwinUNETR", "nnU-Net"]
        found_targets = {t: False for t in targets}
        noise_count = 0
        
        for doc in docs:
            filename = doc.get("file_name", "")
            print(f"- {filename} ({doc['id']})")
            
            for t in targets:
                if t.lower() in filename.lower():
                    found_targets[t] = True
            
            if not any(t.lower() in filename.lower() for t in targets):
                noise_count += 1
        
        print("\nTarget Verification:")
        for t, found in found_targets.items():
            print(f"- {t}: {'FOUND' if found else 'MISSING'}")
            
        print(f"\nNoise Documents: {noise_count} (Need at least 10)")
        
        if all(found_targets.values()) and noise_count >= 10:
            print("\nSUCCESS: Dataset requirements met.")
        else:
            print("\nWARNING: Dataset requirements NOT met.")

    except Exception as e:
        print(f"Error querying database: {e}")

if __name__ == "__main__":
    verify_docs("c1bae279-c099-4c45-ba19-2bb393ca4e4b")
