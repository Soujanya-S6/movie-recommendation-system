import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

def load_data():

    df = pd.read_csv("tmdb_5000_movies.csv")

    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")

    df["combined_text"] = (
        df["overview"] + " " + df["genres"]
    )

    df["clean_text"] = df["combined_text"].str.lower()

    df["clean_text"] = df["clean_text"].apply(
        lambda x: re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            x
        )
    )

    df["clean_text"] = df["clean_text"].apply(
        lambda x: re.sub(
            r"\s+",
            " ",
            x
        ).strip()
    )

    def remove_stopwords(text):
        words = text.split()

        words = [
            word
            for word in words
            if word not in ENGLISH_STOP_WORDS
        ]

        return " ".join(words)

    df["clean_text"] = df["clean_text"].apply(
        remove_stopwords
    )

    return df

def create_model(df):

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(
        df["clean_text"]
    )

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )

    return similarity_matrix

def recommend(item_name, df, similarity_matrix, top_n=5):

    indices = pd.Series(
        df.index,
        index=df["title"]
    ).drop_duplicates()

    if item_name not in indices:
        return []

    idx = indices[item_name]

    similarity_scores = list(
        enumerate(similarity_matrix[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = [
        item
        for item in similarity_scores
        if item[0] != idx
    ]

    top_movies = similarity_scores[:top_n]

    recommendations = []

    for movie_index, score in top_movies:

        recommendations.append({
            "title": df.iloc[movie_index]["title"],
            "score": score
        })

    return recommendations


st.title("Movie Recommendation System")

df = load_data()

similarity_matrix = create_model(df)
movie_list = sorted(
    df["title"].dropna().unique()
)

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)


if st.button("Get Recommendations"):

    recommendations = recommend(
        selected_movie,
        df,
        similarity_matrix,
        5
    )

    st.subheader(
        f"Movies similar to {selected_movie}"
    )

    if recommendations:

        for i, movie in enumerate(
            recommendations,
            start=1
        ):

            st.write(
                f"{i}. {movie['title']}"
            )

    else:

        st.warning(
            "No recommendations found"
        )