import streamlit as st
import pickle
import requests
import numpy as np


api_data = requests.get("https://thronesapi.com/api/v2/Characters").json()


df = pickle.load(open('data.pkl','rb'))
df = df.head(25)

df['characters'] = df['characters'].str.replace('Jaime','Jamie')
df['characters'] = df['characters'].str.replace('Lord Varys','Varys')
df['characters'] = df['characters'].str.replace('Bronn','Lord Bronn')
df['characters'] = df['characters'].str.replace('Sandor Clegane','The Hound')
df['characters'] = df['characters'].str.replace('Robb Stark','Rob Stark')

# Function to fetch image from Thrones API
def fetch_image(name, api_data):
    for item in api_data:
        if item['fullName'] == name:
            return item['imageUrl']
    return "https://via.placeholder.com/300x400.png?text=Image+Not+Found"

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="Game of Thrones Personality Matcher", page_icon="🪑", layout="wide")


st.markdown(
    """
    <h1 style="text-align:center; color:#d62828;">⚔️ Game Of Thrones Personality Matcher ⚔️</h1>
    <p style="text-align:center; font-size:18px; color:#003049;">
        Discover which GOT character is closest to your chosen one!
    </p>
    """, unsafe_allow_html=True
)


st.sidebar.title("About")
st.sidebar.info(
    "This app matches a selected **Game of Thrones character** with the most similar one, "
    "based on vector embeddings. Built with **Streamlit** and Thrones API."
)


characters = df['characters'].values
selected_character = st.selectbox("🔎 Select a character", characters)


character_id = np.where(df['characters'].values == selected_character)[0][0]
x = df[['x','y']].values

distances = []
for i in range(len(x)):
    distances.append(np.linalg.norm(x[character_id] - x[i]))

recommended_id = sorted(list(enumerate(distances)), key=lambda x: x[1])[1][0]
recommended_character = df['characters'].values[recommended_id]


image_url = fetch_image(selected_character, api_data)
recommended_character_image_url = fetch_image(recommended_character, api_data)


similarity_score = round(1 / (1 + distances[recommended_id]) * 100, 2)


def show_fixed_image(url, caption, width=300, height=400):
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="{url}" style="width:{width}px; height:{height}px; object-fit:cover; border-radius:12px;" />
            <p style="font-size:14px; color:gray;">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

col1, col2 = st.columns(2)

with col1:
    show_fixed_image(image_url, f"Selected Character: {selected_character}")

with col2:
    show_fixed_image(recommended_character_image_url, f"Recommended Match: {recommended_character}")



st.markdown(
    f"""
    <div style="text-align:center; margin-top:30px;">
        <h2 style="color:#d62828;">🔥 Similarity Score: {similarity_score}% 🔥</h2>
    </div>
    """,
    unsafe_allow_html=True
)
