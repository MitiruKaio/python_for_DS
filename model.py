from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re
import pandas as pd
import numpy as np
import pickle


def parse_product_name(name):
    pattern = r'((\d+)-комнатная|Студия),\s*([\d,]+)\s*м²'
    match = re.search(pattern, name)
    if match:
        room_type = match.group(1)
        area = float(match.group(3).replace(',', '.'))

        if room_type == 'Студия':
            rooms = 0
        else:
            rooms = int(match.group(2))

        return pd.Series({'rooms': rooms, 'area': area})
    else:
        return pd.Series({'rooms': None, 'area': None})

pd.set_option('display.max_columns', None)

df = pd.read_csv('realty_data.csv')

df[['parsed_rooms', 'parsed_area']] = df['product_name'].apply(parse_product_name)
df = df.drop(['parsed_area', 'period', "city", "district", "area", "address_name", 'settlement', 'description', 'product_name', 'rooms', 'postcode', 'object_type'], axis=1)

df = df.dropna()


df = pd.get_dummies(df, columns=['source'], prefix='source')

y = df['price']
X = df.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_columns = X_train.columns.tolist()
np.save('feature_columns2.npy', feature_columns)

feature_columns = np.load('feature_columns2.npy', allow_pickle=True).tolist()

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE): {mse}")
print(f"Корень из среднеквадратичной ошибки (RMSE): {rmse}")
print(f"Средняя абсолютная ошибка (MAE): {mae}")
print(f"Коэффициент детерминации (R^2): {r2}")


print("\nПрогнозы (первые 10 строк):")
print(pd.DataFrame({'Фактическая цена': y_test, 'Прогнозируемая цена': y_pred}).head(10))

with open('scaler2.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model2.pkl', 'wb') as f:
    pickle.dump(model, f)