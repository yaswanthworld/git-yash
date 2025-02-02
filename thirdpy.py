#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import string

# Load true news dataset
true_news = pd.read_csv("True.csv")

# Load fake news dataset
fake_news = pd.read_csv("Fake.csv")

# Add label column to both datasets
true_news['label'] = 1  # 1 for true news
fake_news['label'] = 0  # 0 for fake news

# Merge the datasets
merged_data = pd.concat([true_news, fake_news])
# Shuffle the merged dataset
merged_data = shuffle(merged_data)

# Reset index
merged_data.reset_index(drop=True, inplace=True)

# Check the shape of the merged dataset
print("Shape of merged dataset:", merged_data.shape)

# Display first few rows of the merged dataset
print(merged_data.head())

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets (75% train, 25% test)
train_data, test_data = train_test_split(merged_data, test_size=0.25, random_state=42)

# Display the shapes of the training and testing sets
print("Shape of training dataset:", train_data.shape)
print("Shape of testing dataset:", test_data.shape)


# In[5]:


# Define function for text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    # Join tokens back into string
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

# Apply text preprocessing to the 'text' column
merged_data['text'] = merged_data['text'].apply(preprocess_text)

# Display first few rows of preprocessed text
print(merged_data['text'].head())


# In[6]:


from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Create an imputer object
imputer = IterativeImputer(max_iter=10, random_state=0)

# Fit the imputer on the data and transform it
scaled_data_imputed = imputer.fit_transform(merged_data)

# Drop rows with NaN values
merged_data.dropna(inplace=True)
# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
scaled_data_imputed = imputer.fit_transform(merged_data)

# Compute the length of the title and text
merged_data['title_length'] = merged_data['title'].apply(len)
merged_data['text_length'] = merged_data['text'].apply(len)

# Count the number of words in the title and text
merged_data['title_word_count'] = merged_data['title'].apply(lambda x: len(x.split()))
merged_data['text_word_count'] = merged_data['text'].apply(lambda x: len(x.split()))

# Compute the average word length in the title and text
merged_data['title_avg_word_length'] = merged_data.apply(lambda row: np.mean([len(word) for word in row['title'].split()]), axis=1)
merged_data['text_avg_word_length'] = merged_data.apply(lambda row: np.mean([len(word) for word in row['text'].split()]), axis=1)

# Count the number of punctuation marks in the title and text
merged_data['title_punctuation_count'] = merged_data['title'].apply(lambda x: sum([1 for char in x if char in string.punctuation]))
merged_data['text_punctuation_count'] = merged_data['text'].apply(lambda x: sum([1 for char in x if char in string.punctuation]))

# Convert binary presence/absence of certain keywords or phrases to numeric (0 or 1)
keywords = ['breaking news', 'exclusive report']
for keyword in keywords:
    merged_data[keyword] = merged_data['text'].apply(lambda x: 1 if keyword in x.lower() else 0)

# Count the frequency of certain words or phrases and convert it to numeric
# Here, you can use CountVectorizer to convert text data into numerical vectors
# You can customize the vectorizer to extract specific words or phrases of interest
vectorizer = CountVectorizer()
text_vectors = vectorizer.fit_transform(merged_data['text'])

# Perform dimensionality reduction using TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
text_vectors_svd = svd.fit_transform(text_vectors)

# Concatenate the numeric features and text vectors
numeric_features = merged_data[['title_length', 'text_length', 'title_word_count', 'text_word_count', 
                                'title_avg_word_length', 'text_avg_word_length', 'title_punctuation_count', 
                                'text_punctuation_count'] + keywords]
cluster_data = pd.concat([numeric_features, pd.DataFrame(text_vectors_svd)], axis=1)

# Convert column names to strings
cluster_data.columns = cluster_data.columns.astype(str)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Determine the optimal number of clusters using silhouette score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(scaled_data, labels))

optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 because we started from k=2

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_data['cluster'] = kmeans.fit_predict(scaled_data)

# Output clusters for the optimal k
print(f"Clusters for k={optimal_k}:")
print(cluster_data['cluster'].value_counts())

# Examine the relationship between formed clusters and the label of the original dataset
cluster_counts = cluster_data.groupby(['cluster', 'label']).size().unstack(fill_value=0)
print("\nCluster counts by label:")
print(cluster_counts)


# In[26]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Drop the 'label' column
cluster_data = merged_data.drop('label', axis=1)

# Choose 10 parameters for clustering
selected_parameters = ['title']  # Add more parameters here as desired

# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
text_vectors = tfidf_vectorizer.fit_transform(cluster_data['title'])

# Apply K-means clustering with k=10
k_values = range(2, 11)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
axes = axes.flatten()

for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_data['cluster'] = kmeans.fit_predict(text_vectors)
    
    # Visualize clusters using TruncatedSVD
    svd = TruncatedSVD(n_components=2)
    text_svd = svd.fit_transform(text_vectors)
    
    # Visualize clusters using TruncatedSVD
    ax = axes[i]
    ax.set_title(f"k = {k}")
    sns.scatterplot(x=text_svd[:, 0], y=text_svd[:, 1], hue=cluster_data['cluster'], ax=ax)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()


# In[ ]:




