import streamlit as st
import pandas as pd
import spacy
import es_core_news_sm
import re
import streamlit as st
import networkx
from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, LabelSet
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
from bokeh.transform import linear_cmap



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

    # Noun chunks
    texto_Palabra = []
    raiz_Palabra = []
    dependencia_con_raiz = []
    texto_raiz_conectada = []

    for chunk in doc.noun_chunks:
        texto_Palabra.append(chunk.text)
        raiz_Palabra.append(chunk.root.text)
        dependencia_con_raiz.append(chunk.root.dep_)
        texto_raiz_conectada.append(chunk.root.head.text)
    
    df_Palabra = pd.DataFrame({'Palabra': texto_Palabra, 'Raiz':raiz_Palabra, 'Relación de dependencia con raíz': dependencia_con_raiz, 'Nucleo': texto_raiz_conectada, 'Peso': 1}) 

    ## GRAFO ##
    G = networkx.from_pandas_edgelist(df_Palabra, 'Nucleo', 'Palabra', 'Peso')
    
    degrees = dict(networkx.degree(G))
    networkx.set_node_attributes(G, name='degree', values=degrees)

    number_to_adjust_by = 15
    adjusted_node_size = dict([(node, degree+number_to_adjust_by) for node, degree in networkx.degree(G)])
    networkx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)
    size_by_this_attribute = 'adjusted_node_size'
    color_by_this_attribute = 'adjusted_node_size'
    color_palette = Blues8


    title = 'Conocimiento extraído del texto'

    #Tooltips
    HOVER_TOOLTIPS = [("Palabra", "@index"),
        ("Degree", "@degree")]

    #Atributos de plot de Bokeh
    plot = figure(tooltips = HOVER_TOOLTIPS,
                tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=None, plot_width=420)

    #Crear grafo
    network_graph = from_networkx(G, networkx.spring_layout, scale=10, center=(0, 0))
    minimum_value_color = min(network_graph.node_renderer.data_source.data[color_by_this_attribute])
    maximum_value_color = max(network_graph.node_renderer.data_source.data[color_by_this_attribute])

    #Tamaño de nodos
    network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=linear_cmap(color_by_this_attribute, color_palette, minimum_value_color, maximum_value_color))

    #Opacidad
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)

    #Agregar grafo
    plot.renderers.append(network_graph)

    #Agregar etiquetas
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='13px', background_fill_alpha=0.8)
    plot.renderers.append(labels)

    col1, col2 = st.columns((1.5,2))

    with col1:
        st.header("Entidades reconocidas")
        st.write(df)
        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')


        csv = convert_df(df)

        st.download_button(
        "Descargar entidades",
        csv,
        "entidades.csv",
        "text/csv",
        key='download-csv'
        )

    with col2:
        st.header("Grafo de conocimiento")
        st.write(plot)
        
