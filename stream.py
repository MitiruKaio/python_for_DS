import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_folium import st_folium
import folium


with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Отметьте точку на карте")

# Инициализация состояния для хранения координат
if 'coords' not in st.session_state:
    st.session_state.coords = None

# Функция для создания карты
def create_map():
    southwest = [55.0, 37.0]
    northeast = [56.0, 38.0]
    center_location = [(southwest[0] + northeast[0]) / 2, (southwest[1] + northeast[1]) / 2]
    m = folium.Map(
        location=center_location,
        zoom_start=10,
        max_bounds=True,
        min_zoom=8,
        max_zoom=15
    )
    m.fit_bounds([southwest, northeast])
    m.options['maxBounds'] = [southwest, northeast]
    folium.TileLayer('openstreetmap').add_to(m)
    folium.LatLngPopup().add_to(m)
    return m

# Создание и отображение карты
m = create_map()
output = st_folium(m, width=700, height=500)

# Проверка, был ли произведен клик и сохранение координат
if output and 'last_clicked' in output and output['last_clicked']:
    lat = output['last_clicked']['lat']
    lon = output['last_clicked']['lng']
    st.session_state.coords = (lat, lon)
    st.success(f"Вы выбрали точку:\n- **Широта:** {lat}\n- **Долгота:** {lon}")

    selected_lat, selected_lng = st.session_state.coords
else:
    st.info("Пожалуйста, кликните на карту, чтобы выбрать точку.")

st.title("Прогнозирование стоимости недвижимости")
st.header("Введите параметры недвижимости")

total_square = st.number_input("Общая площадь (м²)", min_value=10.0, max_value=300.0, value=50.0, step=1.0)
floor = st.number_input("Этаж", min_value=1, max_value=50, value=5, step=1)
parsed_rooms = st.number_input("Количество комнат (0 для студии)", min_value=0, max_value=10, value=2, step=1)

st.subheader("Выберите источник")
source = st.selectbox("Источник", options=['Домклик', 'Новострой-М', 'ЦИАН', 'Яндекс.Недвижимость'])

if st.button("Предсказать стоимость"):
    input_data = pd.DataFrame({
        'lat': [lat],
        'lon': [lon],
        'total_square': [total_square],
        'floor': [floor],
        'parsed_rooms': [parsed_rooms],
        'source': [source],
    })

    input_data = pd.get_dummies(input_data, columns=['source'], prefix='source')

    feature_columns = np.load('feature_columns2.npy', allow_pickle=True).tolist()
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"Прогнозируемая стоимость: {int(prediction[0]):,} рублей")
