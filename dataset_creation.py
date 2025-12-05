import pandas as pd
import re


#Pre-Processing
# The script loads raw data from the CSV file into a data table.
# It counts the articles to see if any category (like "Sport") is unfairly larger than others.
# To fix this imbalance, it randomly removes extra articles until all categories are equal.
# It cleans the text by removing web links, numbers, and special punctuation.
# It merges the article title and body into a single, simple text format for the AI.
# Finally, it saves this balanced and scrubbed data to a new file, processed_data.csv.

# Load data
input_file = "sensed_data.csv"
df = pd.read_csv(input_file)

print(f"Original data shape: {df.shape}")

# --- Step 1: Class Imbalance Analysis & Fix ---
print("\n--- Class Distribution (Before) ---")
label_counts = df['label'].value_counts()
print(label_counts)

min_class_count = label_counts.min()
max_class_count = label_counts.max()

# Rule: If the largest class is more than 2x the smallest, we balance it.
if max_class_count > 2 * min_class_count:
    print("\n[!] Imbalance detected. Performing Down-sampling...")
    
    # Group data by label
    groups = df.groupby('label')
    
    balanced_dfs = []
    for label, group_df in groups:
        # Sample only 'min_class_count' items from each group
        # random_state=42 ensures we get the same random rows every time we run it
        downsampled_group = group_df.sample(n=min_class_count, random_state=42)
        balanced_dfs.append(downsampled_group)
        
    # Recombine the data
    df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"New data shape after balancing: {df.shape}")
    print(df['label'].value_counts())
else:
    print("Data is relatively balanced. No action taken.")

# --- Step 2: Text Cleaning ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation/numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

df['clean_title'] = df['title'].apply(clean_text)
df['clean_body'] = df['body'].apply(clean_text)
df['combined_text'] = df['clean_title'] + " " + df['clean_body']

# --- Step 3: Save ---
output_file = "processed_data.csv"
df.to_csv(output_file, index=False)
print(f"\nSuccess! Processed data saved to: {output_file}")


