from datasets import load_dataset
import pandas as pd

# Step 1: Download the dataset from Hugging Face
# This streams the dataset so we don't download more than we need
print("Downloading dataset from Hugging Face...")
dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

# Step 2: Inspect the first 3 rows to understand the structure
print("\nFirst 3 rows:")
for i in range(3):
    print(f"\n--- Row {i+1} ---")
    print(dataset[i])

# Step 3: Convert the first 500 rows to a pandas DataFrame
print("\nExtracting first 500 rows...")
df = dataset.select(range(500)).to_pandas()

# Step 4: Save the DataFrame as a CSV file
output_path = "data/tickets.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved 500 rows to {output_path}")
print(f"Columns: {list(df.columns)}")
print(f"Shape: {df.shape}")
