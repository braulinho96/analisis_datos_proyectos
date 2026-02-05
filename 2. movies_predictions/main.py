import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error

# Configuración de estilo
sns.set_theme(style="whitegrid")

# CARGA
df_movies = pd.read_csv('movies_predictions/csv/movies.csv')
df_themes = pd.read_csv('movies_predictions/csv/themes.csv')
themes_grouped = df_themes.groupby('id')['theme'].apply(lambda x: ' '.join(x)).reset_index()
themes_grouped.columns = ['id', 'all_themes']
df = pd.merge(df_movies, themes_grouped, on='id', how='left')

# LIMPIEZA Y VALORES FALTANTES
print("\n--- 2. LIMPIEZA ---")
df = df.dropna(subset=['rating'])
df['all_themes'] = df['all_themes'].fillna('unknown')
df = df.dropna(subset=['minute', 'date'])
df = df[df['minute'] > 0]

# DETECCIÓN DE OUTLIERS (ATÍPICOS) 
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['minute'], color='orange')
plt.title('Detección de Outliers en Duración (Antes de limpiar)')
plt.show()
plt.close()

# Regla del IQR: Todo lo que esté muy lejos del promedio se va
Q1 = df['minute'].quantile(0.25)
Q3 = df['minute'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

df_clean = df[(df['minute'] >= limite_inferior) & (df['minute'] <= limite_superior)].copy()
print(f"Filas tras eliminar outliers de duración: {len(df_clean)} (Se eliminaron {len(df) - len(df_clean)})")

# ANÁLISIS DE NORMALIDAD Y TRANSFORMACIÓN
# distribuye normalmente la variable objetivo (duración)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_clean['minute'], kde=True)
plt.title('Distribución Original (Duración)')

# Aplicamos Log-Transformación para normalizar
# Esto ayuda a que el modelo no se obsesione con pelis extremadamente largas o cortas
df_clean['log_minute'] = np.log1p(df_clean['minute'])

# CORRELACIÓN
# Veamos si las variables numéricas se hablan entre sí
plt.figure(figsize=(6, 5))
corr_matrix = df_clean[['rating', 'date', 'minute']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Mapa de Correlación')
plt.show()
plt.close()

# ---------------------------------------------------------
# FEATURE ENGINEERING & SELECTION
# ---------------------------------------------------------
print("\n--- PROCESAMIENTO Y SELECCIÓN ---")

# A. Variables Numéricas (Usamos log_minute en vez de minute)
X_num = df_clean[['date', 'log_minute']].values

# B. Variables de Texto (TF-IDF)
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
X_text = tfidf.fit_transform(df_clean['all_themes'])

# C. SELECCIÓN DE CARACTERÍSTICAS (Nuevo paso)
# De los 2000 temas, ¿cuáles realmente ayudan a predecir el rating?
# Usamos f_regression para ver la correlación entre cada palabra y el rating
selector = SelectKBest(score_func=f_regression, k=300) # Nos quedamos con los 300 mejores temas
X_text_selected = selector.fit_transform(X_text, df_clean['rating'])

print(f"Reducción de dimensiones de texto: De {X_text.shape[1]} a {X_text_selected.shape[1]} features.")

# D. Unir todo
X_final = sp.hstack((X_num, X_text_selected))
y = df_clean['rating']

# ---------------------------------------------------------
# 7. ENTRENAMIENTO Y COMPARACIÓN
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"\n>>> RMSE FINAL MEJORADO: {rmse:.4f} <<<")

# Ver qué temas sobrevivieron y son importantes
# (Un poco de magia para recuperar los nombres después del SelectKBest)
mask = selector.get_support() # True/False de qué columnas quedaron
feature_names = np.array(tfidf.get_feature_names_out())[mask]

# Importancia según el Random Forest (Ojo: los primeros 2 son date y log_minute)
importances = model.feature_importances_
# Indices 0 y 1 son numéricos, del 2 en adelante son texto
print("\nTop 3 Temas que más influyen en el Rating:")
# Ajustamos índices para saltar las numéricas
text_importances = importances[2:] 
top_indices = text_importances.argsort()[-3:][::-1]
for idx in top_indices:
    print(f"- {feature_names[idx]}")