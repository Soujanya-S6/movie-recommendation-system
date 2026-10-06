# Movie Recommendation System

## Project Overview

This project implements a content-based movie recommendation system using the TMDB 5000 Movies Dataset.

The system recommends movies based on the similarity between movie descriptions and genres.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Streamlit
- Git
- GitHub
- Render

## Dataset

TMDB 5000 Movies Dataset obtained from Kaggle. https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

## Recommendation Approach

The recommendation system follows these steps:

1. Load the movie dataset
2. Handle missing values
3. Clean and preprocess movie text
4. Combine movie overview and genre information
5. Convert text into TF-IDF vectors
6. Calculate cosine similarity
7. Recommend the most similar movies

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
