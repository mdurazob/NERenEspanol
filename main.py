import streamlit as st
import pandas as pd
import spacy
import es_core_news_sm
import streamlit as st

st.title("NER en español")

article = st.text_area("Introduce el texto a analizar:", height=200)

nlp = spacy.load('es_core_news_sm')

if st.button("Analizar ahora"):
    doc = nlp(article)
    texto = []
    etiquetas = []
    for ent in doc.ents:
        texto.append(ent.text)
        etiquetas.append(ent.label_)

    df = pd.DataFrame({'Nombre': texto, 'Entidad':etiquetas}) 
    st.write(df)

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    csv = convert_df(df)

    st.download_button(
    "Descargar",
    csv,
    "entidades.csv",
    "text/csv",
    key='download-csv'
    )
