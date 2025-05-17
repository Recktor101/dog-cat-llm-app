from transformers import pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_generator()

def get_breed_description(breed):
    prompt = f"Give a short and friendly description about the {breed} dog breed."
    result = generator(prompt, max_length=50)[0]['generated_text']
    return result
