# para executar:
# windows: python -m streamlit run Projeto-BigData-STREAMLIT.py

import streamlit as st
# Importando as bibliotecas pandas e numpy, importante para tratamento e visualização dos dados importados
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df_geolocation = pd.read_csv('csv/olist_geolocation_dataset.csv')
df_customers = pd.read_csv('csv/olist_customers_dataset.csv')
df_items = pd.read_csv('csv/olist_order_items_dataset.csv')
df_payments = pd.read_csv('csv/olist_order_payments_dataset.csv')
df_reviews = pd.read_csv('csv/olist_order_reviews_dataset.csv')
df_orders = pd.read_csv('csv/olist_orders_dataset.csv')
df_products = pd.read_csv('csv/olist_products_dataset.csv')
df_sellers = pd.read_csv('csv/olist_sellers_dataset.csv')
df_names = pd.read_csv('csv/product_category_name_translation.csv')

st.title('Projeto Ecossistema de Big Data')

st.markdown('No presente projeto, utilizou-se o Brazilian E-Commerce Public Dataset by Olist como base para simular o papel de engenheiros de dados em uma empresa de e-commerce. O propósito é extrair insights valiosos que não apenas melhorem as operações de negócios, mas também otimizem a logística e aprimorem a experiência do cliente. Ao mergulharmos nos dados reais do setor de e-commerce brasileiro, esta iniciativa oferece uma oportunidade prática de explorar tendências, identificar padrões e, consequentemente, orientar decisões estratégicas. Além disso, destacamos a importância de análises orientadas por dados no contexto dinâmico do comércio eletrônico, evidenciando como tais abordagens podem impulsionar melhorias tangíveis e sustentáveis no desempenho empresarial.')

st.subheader('Como os dados podem ser usados para melhorar a recomendação de produtos e personalizar a experiência do cliente?')
st.write('Utilizando o dataset da Olist podemos aplicar algoritmos  de Machine Learning para analisar o comportamento de compra. Podemos usar os reviews para subcategorizar compradores de acordo com perfis de consumo semelhanntes e indicar produtos.')

merged_df = pd.merge(df_products, df_items, on='product_id')

category_counts = merged_df['product_category_name'].value_counts().reset_index()
category_counts.columns = ['category', 'count']

fig = px.bar(category_counts, x='category', y='count',
             title='Contagem de Pedidos por Categoria',
             labels={'category': 'Categoria', 'count': 'Número de Pedidos'})
st.plotly_chart(fig, use_container_width=True)


st.subheader('Quais insights podem ser obtidos para otimizar a gestão de inventário e operações logísticas?')
st.write('Análise preditiva pode ser utilizada para prever a demanda de produtos e otimizar o estoque. Técnicas de Machine Learning podem ser utilizadas para identificar padrões de compra de acordo com a época do ano (Séries Temporais), para ajudar a prever períodos de alta demanda e necessidade de reposição de estoque. Além disso os dados de geolocation, podem ajudar a otimizar rotas de entrega e otimizar custos e dimunuir prazos.')

df_geolocation['order_count'] = df_geolocation.groupby('geolocation_zip_code_prefix')['geolocation_zip_code_prefix'].transform('count')


fig = px.density_mapbox(df_geolocation, lat='geolocation_lat', lon='geolocation_lng', z='order_count', radius=10,
                        center=dict(lat=-21.176, lon=-46.390), zoom=3,
                        mapbox_style="stamen-terrain",
                        title='Mapa de Calor de Pedidos por Geolocalização')
st.plotly_chart(fig, use_container_width=True)

st.subheader('De que maneira a visualização de dados pode auxiliar na decisão estratégica e operacional?')
st.write('Visualizações de heatmaps, gráficos de linha e dashboards interativos pode oferecer insights rápidos e claros para tomada de decisão.')


merged_df = pd.merge(df_orders, df_customers, on='customer_id')


state_order_counts = merged_df.groupby('customer_state')['order_id'].nunique().reset_index()
state_order_counts.columns = ['state', 'order_count']


fig = px.density_heatmap(state_order_counts, x='state', y='order_count',
                         nbinsx=50, title='Heatmap de Tendências de Compras por Estado',
                         labels={'state': 'Estado', 'order_count': 'Contagem de Pedidos'})
fig.update_traces(xbins=dict( # update the number of bins
    start=0.5,
    end=state_order_counts['state'].nunique() + 0.5,
    size=1
))
st.plotly_chart(fig, use_container_width=True)