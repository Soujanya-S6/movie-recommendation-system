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

    return tfidf_matrix

def recommend(item_name, df, tfidf_matrix, top_n=5):

    indices = pd.Series(
        df.index,
        index=df["title"]
    ).drop_duplicates()

    if item_name not in indices:
        return []

    idx = indices[item_name]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    recommendations = []

    for movie_index in similar_indices:

        if movie_index == idx:
            continue

        recommendations.append({
            "title": df.iloc[movie_index]["title"],
            "score": similarity_scores[movie_index]
        })

        if len(recommendations) == top_n:
            break

    return recommendations


st.title("Movie Recommendation System")

df = load_data()

tfidf_matrix = create_model(df)
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
        tfidf_matrix,
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