import streamlit as st # pyright: ignore[reportMissingImports]
import pickle
import joblib # pyright: ignore[reportMissingImports]

st.title("Movie Recommendation System")
with open("movies.pickle",'rb') as m:
    movies = pickle.load(m)

similarities=joblib.load("similarity.joblib",'rb')

movie_names = movies['title'].values

def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]
    recommendations = similarities[movie_index]

    movie_list=sorted(enumerate(recommendations), reverse = True, key= lambda x: x[1])[1:6]
    
    recommended_movies = []
    
    for i in movie_list:

        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies



movie_name=st.selectbox("Enter the Movie Name: ",movie_names)

if st.button("Recommend"):
    st.write("The Recommended Movies are: ")
    for i in recommend(movie_name):
        st.write(i)
