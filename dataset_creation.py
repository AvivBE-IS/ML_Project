import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

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

#Segmentation
# --- Step 2.5: Segmentation ---
# The following recommended steps for text labeling have already been performed:
# 1. Field Segmentation: Completed (in sensing.py).
# 2. Word Normalization: Completed (in dataset_creation.py).
# 3. Model Input: The full 'combined_text' is passed to the model.
#    (This is the standard approach for document classification).


# --- Step 3: Feature Extraction (Bag of Words) ---
# It acts as a translator that turns English text into Math.
# The Vocabulary: First, the code scanned all your articles to find the 1,000 most common words (like "election", "game", "court") to create a master list.
# The Counting: It then revisited every single article and counted exactly how many times those specific words appeared in the text.
# The Result: Instead of sentences, you now have a grid of numbers.
# If an article is about sports, the column for the word "ball" might have a 5.
# If an article is about politics, the column for the word "ball" will likely have a 0.
print("\n--- Feature Extraction ---")

# 1. Build Vocabulary (Find top 1000 most common words)
# We join all text into one giant string, split it, and count the words
all_words = pd.Series(' '.join(df['combined_text']).split())
vocabulary = all_words.value_counts().head(1000).index.tolist()

print(f"Vocabulary size: {len(vocabulary)} words")

# 2. Create Feature Matrix
# We count how often the vocabulary words appear in each specific article
def count_features(text):
    # Count all words in this specific text
    word_counts = Counter(text.split())
    # Only keep the counts for words that are in our main vocabulary
    return {word: word_counts.get(word, 0) for word in vocabulary}

# Apply this to every row
features_list = df['combined_text'].apply(count_features).tolist()
features_df = pd.DataFrame(features_list)

# Fill empty spots with 0 (meaning the word didn't appear in that article)
features_df = features_df.fillna(0).astype(int)

# Add the label back so the model knows the answer
features_df['label'] = df['label']

print(f"Features Matrix shape: {features_df.shape}")

# --- Step 4: Save ---
output_file = "processed_data.csv"
df.to_csv(output_file, index=False)

feature_file = "features_data.csv"
features_df.to_csv(feature_file, index=False)

print(f"\nSuccess! Processed data saved to: {output_file}")
print(f"Success! Numerical features saved to: {feature_file}")

# --- Step 3: Feature Representation ---
print("\n--- Advanced Feature Representation: TF-IDF ---")

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['combined_text'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

tfidf_df['label'] = df['label'].values

tfidf_file = "tfidf_features.csv"
tfidf_df.to_csv(tfidf_file, index=False)

print(tfidf_df.head().to_string())
print(f"\nSuccess! TF-IDF features saved to: {tfidf_file}")
# --- Step 4: Feature Selection ---
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

print("\n--- Step 4: Feature Selection (Chi-Square Filter) ---")

# Separate features (X) and target label (y)
X = tfidf_df.drop('label', axis=1)
y = tfidf_df['label']

print(f"Original Feature Count: {X.shape[1]}")

# Initialize Chi-Square selector to keep top 1000 features
k_best_selector = SelectKBest(score_func=chi2, k=1000)
X_new = k_best_selector.fit_transform(X, y)

# Get the names of the selected features
selected_mask = k_best_selector.get_support()
selected_feature_names = X.columns[selected_mask]

# Create a new DataFrame with only the selected features
filtered_df = pd.DataFrame(X_new, columns=selected_feature_names)
filtered_df['label'] = y.values

print(f"New Feature Count: {filtered_df.shape[1]-1}")

# Save the filtered dataset
filtered_file = "selected_features.csv"
filtered_df.to_csv(filtered_file, index=False)

print(f"Selected features saved to: {filtered_file}")

# Display the top 10 most informative features
scores = pd.DataFrame({'Feature': X.columns, 'Score': k_best_selector.scores_})
print("\nTop 10 Most Informative Features:")
print(scores.sort_values(by='Score', ascending=False).head(10).to_string(index=False))


# --- Step 5: Dimensionality Reduction ---


print("\n--- Step 5: Dimensionality Reduction (PCA) & Visualization ---")

# 1. Prepare the data
# We use the filtered dataset from Step 5 (Feature Selection)
# 'filtered_df' should be in memory. If not, uncomment the lines below to load it:
# filtered_df = pd.read_csv("selected_features.csv")

# Separate features (X) and target label (y)
X_for_pca = filtered_df.drop('label', axis=1)
y_for_pca = filtered_df['label']

# 2. Perform PCA - Reduce to 3 dimensions
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_for_pca)

# 3. Create a DataFrame for the results
# This makes plotting easier
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
pca_df['label'] = y_for_pca.values

print(f"Data reduced from {X_for_pca.shape[1]} dimensions to 3 dimensions.")

# 4. Create a 3D Scatter Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Map labels to colors (optional, for better customization)
# Adjust the keys ('Benign', 'Malicious') to match your actual data labels
colors = {'Benign': 'blue', 'Malicious': 'red'} 

# Plot the points
# We factorize the label to convert string labels to numbers for coloring
sc = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], 
                c=pd.factorize(pca_df['label'])[0], 
                cmap='viridis', # Colormap
                s=50,           # Dot size
                alpha=0.6)      # Transparency

# Set axis labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D Visualization of Malware vs Benign (PCA)')

# Add a legend
# Note: This automatically creates a legend based on the unique values in 'c'
plt.legend(*sc.legend_elements(), title="Classes")

# Show the plot
plt.show()

# 5. Save the reduced dataset (Optional)
# This file can be used later for training models on the reduced data
pca_df.to_csv("pca_3d_dataset.csv", index=False)
print("PCA dataset saved to: pca_3d_dataset.csv")



# --- Step 6: Validation ---
print("\n--- Step 6: Validation (Logistic Regression with K-Fold) ---")

# 1. Prepare Data
# We use the dataset from Step 5 (Feature Selection) which contains the top 1000 features.
# If 'filtered_df' is not in memory, uncomment the next line to load it:
# filtered_df = pd.read_csv('selected_features.csv')

# Separate features (X) and target label (y)
X = filtered_df.drop('label', axis=1)
y = filtered_df['label']

# 2. Initialize the Model
# We use Logistic Regression as requested.
# max_iter=1000 is set to ensure the solver converges on this dataset size.
model = LogisticRegression(max_iter=1000)

# 3. Define Validation Strategy
# StratifiedKFold ensures each fold preserves the percentage of samples for each class.
# n_splits=10: The standard 10-fold cross-validation.
# shuffle=True: Randomize the data before splitting.
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 4. Define Evaluation Metric
# We use AUC (Area Under Curve) because it is robust for classification tasks.
metric = 'roc_auc'

# 5. Perform Validation
# This runs the training and testing process 10 times.
print("Running Cross-Validation... this might take a moment.")
scores = cross_val_score(model, X, y, cv=cv, scoring=metric)

# 6. Display Results
print(f"Validation Method: 10-Fold Stratified Cross-Validation")
print(f"Chosen Metric: {metric}")
print(f"\nScores for each fold: \n{scores}")
print(f"\n>>> Average {metric} Score: {scores.mean():.4f}")