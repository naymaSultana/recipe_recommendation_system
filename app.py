import streamlit as st
import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import re
import pandas as pd

from train import model, words, labels, data
#Download NLTK data
#nltk.download('punkt')

with open('/Users/alohomora/Downloads/combined/recipes.json') as f:
    data = json.load(f)


stemmer = SnowballStemmer("english")





def bag_of_words(s, words, stemmer):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat(inp):
    results = model.predict([bag_of_words(inp, words, stemmer)])[0]
    results_index = np.argmax(results)
    recipe_name = labels[results_index]

    if results[results_index] > 0.05:
        for recipe in data["recipes"]:
            if recipe["recipe_name"] == recipe_name:
                ner = ", ".join(recipe["NER"])
                steps = recipe["steps"]
                ingredients = recipe["ingredients"]
                link = recipe["link"]
                return recipe_name, ner, steps, ingredients, link
    else:
        return None, None, None, None, None

def display_recipe_card(recipe_name, ner, steps, ingredients, link):
   
    ner_elements = [f"• {element.capitalize()}" for element in ner.split(", ")]
    ner_bullet_points = "<br>".join(ner_elements)
    if not link.startswith("https://"):
        link = "https://" + link

    st.markdown(
        f"""
        <div style="background: linear-gradient(to bottom right, #212121, #000000);padding:20px;border-radius:10px;box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);border: 2px solid #ffa79d;">
            <h2 style="color:#ffffff; text-shadow: 0 0 5px #FFE6E6;">{recipe_name}</h2>
            <h3 style="color:#ffffff;">Ingredients:</h3>
            <p style="color:#BDBDBD;">{ner_bullet_points}</p>
            <h3 style="color:#ffffff;">Portions and Quantities:</h3>
            {ingredients_table(ingredients)}
            <h3 style="color:#ffffff;">Steps:</h3>
            <ol style="color:#BDBDBD;">
                {"".join([f"<li>{step}</li>" for step in steps])}
            </ol>
            <a href="{link}" style="color:#ff7464;text-shadow: 0 0 8px #ca5d5d;text-decoration:none;" target="_blank">Link to recipe</a>
        </div>
        """,
        unsafe_allow_html=True
    )


def ingredients_table(ingredients):
    df_ingredients = pd.DataFrame(ingredients, columns=["Ingredients"])
    table_style = """
        color:#BDBDBD;
        border-collapse: collapse;
        width: 100%;
    """
    th_td_style = """
        border: 1px solid #555;
        padding: 8px;
        text-align: left;
    """
    return df_ingredients.to_html(index=False, justify='left', classes='ingredients-table', table_id='ingredients-table', escape=False, header=False, col_space=10, border=0, max_rows=None).replace('<table', f'<table style="{table_style}"').replace('<th', f'<th style="{th_td_style}"').replace('<td', f'<td style="{th_td_style}"')

st.title("Recipe Recommender")
ingredient_input = st.text_input("Enter ingredients:", "")
if st.button("Get Recipe"):
    recipe_name, ner, steps, ingredients, link = chat(ingredient_input)
    if recipe_name:
        display_recipe_card(recipe_name, ner, steps, ingredients, link)
    else:
        st.write("Sorry, I couldn't find a recipe for those ingredients.")
