import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
data = {
    'Movie': ['Inception', 'Interstellar', 'The Dark Knight', 'Avengers', 'Titanic'],
    'Genre': [
        'Sci-Fi Thriller',
        'Sci-Fi Drama',
        'Action Crime',
        'Action Sci-Fi',
        'Romance Drama'
    ]
}
df = pd.DataFrame(data)
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df['Genre'])
similarity = cosine_similarity(genre_matrix)
def recommend(movie_name):
    if movie_name not in df['Movie'].values:
        return "Movie not found in database."
    index = df[df['Movie'] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\nMovies recommended for '{movie_name}':")
    for i in scores[1:4]:
        print(df.iloc[i[0]]['Movie'])
recommend("Inception")
