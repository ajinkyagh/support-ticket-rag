import os
import pandas as pd
import ollama
from supabase import create_client
from dotenv import load_dotenv

# Step 1: Load environment variables from .env file
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Step 2: Connect to Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Step 3: Load the CSV file
print("Loading tickets from CSV...")
df = pd.read_csv("data/tickets_augmented.csv")
total = len(df)
print(f"Loaded {total} tickets")

# Step 4: Loop through each row, generate embedding, and insert into Supabase
success_count = 0
skip_count = 0

for idx, row in df.iterrows():
    try:
        # Generate embedding using Ollama's nomic-embed-text model (768 dimensions)
        result = ollama.embeddings(model="nomic-embed-text", prompt=row["instruction"])
        embedding = result["embedding"]

        # Build the record to insert
        record = {
            "ticket_id": str(idx),
            "subject": str(row["instruction"])[:50],
            "body": str(row["instruction"]),
            "category": str(row["category"]),
            "priority": str(row["intent"]),  # using intent as priority
            "embedding": embedding,
        }

        # Insert the record into the Supabase tickets table
        supabase.table("tickets").insert(record).execute()
        success_count += 1

        # Print progress every 50 tickets
        if success_count % 50 == 0:
            print(f"Embedded {success_count}/900")

    except Exception as e:
        # Skip failed rows and continue processing the rest
        skip_count += 1
        print(f"Skipping row {idx}: {e}")

print(f"\nDone. Inserted {success_count} tickets, skipped {skip_count}.")
