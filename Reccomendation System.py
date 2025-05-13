import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample dataset
data = {
    'User': ['User1', 'User1', 'User2', 'User2', 'User3', 'User3'],
    'Item': ['ItemA', 'ItemB', 'ItemA', 'ItemC', 'ItemB', 'ItemC'],
    'Rating': [5, 3, 4, 2, 5, 4]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a pivot table
pivot_table = df.pivot_table(index='User', columns='Item', values='Rating').fillna(0)

# Compute cosine similarity
similarity_matrix = cosine_similarity(pivot_table)

# Create a DataFrame for similarity matrix
similarity_df = pd.DataFrame(similarity_matrix, index=pivot_table.index, columns=pivot_table.index)

# Function to recommend items
def recommend(user, pivot_table, similarity_df, num_recommendations=2):
    user_ratings = pivot_table.loc[user]
    similar_users = similarity_df[user].sort_values(ascending=False).index[1:]
    
    recommendations = {}
    for similar_user in similar_users:
        similar_user_ratings = pivot_table.loc[similar_user]
        for item, rating in similar_user_ratings.items():
            if user_ratings[item] == 0 and rating > 0:
                recommendations[item] = recommendations.get(item, 0) + rating
    
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in sorted_recommendations[:num_recommendations]]

# Example usage
user_to_recommend = 'User1'
recommended_items = recommend(user_to_recommend, pivot_table, similarity_df)
print(f"Recommended items for {user_to_recommend}: {recommended_items}")