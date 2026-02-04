import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB


nltk.download('stopwords')

# Funcion que elimina las palabras irrelevantes y caracteres especiales
def limpiar_texto(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)

# ---------------------------------------------------------
# Cargamos los datos
df_fakes = pd.read_csv('fake_news/Fake.csv').dropna()
df_trues = pd.read_csv('fake_news/True.csv').dropna()

# Evaluamos el balance de clases, para evaluar si es necesario aplicar técnicas de balanceo
relacion = len(df_trues) / len(df_fakes)
print(f"Relacion Verdaderas/Falsas: {relacion}")

# Agregamos la columna de etiquetas
df_fakes['label'] = 0
df_trues['label'] = 1
data = pd.concat([df_fakes, df_trues], ignore_index=True)
df = pd.DataFrame(data)
df = df[['text', 'label']]  # Seleccionamos solo las columnas necesarias

# Limpieza de texto 
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['text'].apply(limpiar_texto)

# Vectorizacion TF-IDF
# Convertimos el texto en una matriz de números y separamos las etiquetas
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text']) 
y = df['label']

# Seleccionamos conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# - - - - - - -  modelo Regresión Logística - - - - - - - -
model_regresion = LogisticRegression(
    class_weight='balanced'
    )
model_regresion.fit(X_train, y_train)

# - - - - - - - modelo de Naive-Bayes - - - - - - - - - - -
model_bayes = MultinomialNB()
model_bayes.fit(X_train, y_train)

# Evaluamos los modelos
prediction = model_regresion.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

prediction_bayes = model_bayes.predict(X_test)
accuracy_bayes = accuracy_score(y_test, prediction_bayes)

print("\nResultados:")
print("\nMatriz de Confusión regresión:")
print(confusion_matrix(y_test, prediction))
print(f"Precisión del modelo de regresión: {accuracy * 100:.2f}%")
print("\nMatriz de Confusión Bayes:")
print(confusion_matrix(y_test, prediction_bayes))
print(f"\nPrecisión del modelo de Bayes: {accuracy_bayes * 100:.2f}%")
print("\n")


print("--- Análisis de Regresión Logística ---")

# Obtenemos el diccionario de palabras
feature_names = vectorizer.get_feature_names_out()

# Obtenemos los coeficientes o pesos, luego del entrenamiento
coefs = model_regresion.coef_[0].argsort()

# Las palabras con peso mas negativo estan asociadas a FAKE - Clase 0 y el mas positivo a TRUE - Clase 1
print("\nTop 10 palabras que indican NOTICIA FAKE:")
top_fake_reg = [feature_names[i] for i in coefs[:10]]
print(top_fake_reg)
print("\nTop 10 palabras que indican NOTICIA TRUE:")
top_true_reg = [feature_names[i] for i in coefs[-10:]]
print(top_true_reg)

# - - - - - - - - - - - - - - -

print("\n\n--- Analisis de Naive Bayes ---")

# feature_log_prob_ es una matriz de tamaño (2, n_palabras)
# Fila 0: Log-probabilidad de cada palabra dado que es FAKE
log_prob_fake = model_bayes.feature_log_prob_[0]
log_prob_true = model_bayes.feature_log_prob_[1]

# Calculamos qué palabras son mucho más probables en Fake que en True (Diferencia grande a favor de Fake)
diferencia = log_prob_fake - log_prob_true
sorted_bayes_indices = diferencia.argsort()

# Las palabras donde la probabilidad de Fake gana por goleada (índices altos en la diferencia)
print("\nTop 10 palabras más 'Bayesianas' para FAKE:")
top_fake_bayes = [feature_names[i] for i in sorted_bayes_indices[-10:]]
print(top_fake_bayes)
print("\nTop 10 palabras más 'Bayesianas' para TRUE:")
top_true_bayes = [feature_names[i] for i in sorted_bayes_indices[:10]]
print(top_true_bayes)

