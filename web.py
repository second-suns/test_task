import streamlit as st
from finder import Finder


@st.cache_resource
def load_model():
    f = Finder()
    return f


sentences = st.text_input("Поисковый запрос")

if st.button('Submit'):
    a = load_model().search(sentences.title())
    for i in a:
        st.markdown(f"[{i[0]}]({i[1]})")

load_model()
