from supabase_client import supabase

def check_columns():
    try:
        # Select * with limit 1 to see the columns in the response data keys
        response = supabase.table("conversations").select("*").limit(1).execute()
        if response.data:
            print(f"Columns in 'conversations': {list(response.data[0].keys())}")
        else:
            # If no data, we might need another way or just assume it's empty
            print("No data in 'conversations' table to check columns via select *.")
            # Try selecting metadata explicitly
            try:
                supabase.table("conversations").select("metadata").limit(1).execute()
                print("Column 'metadata' exists (explicit select succeeded).")
            except Exception as e:
                print(f"Column 'metadata' likely missing: {e}")
    except Exception as e:
        print(f"Error checking columns: {e}")

if __name__ == "__main__":
    check_columns()
