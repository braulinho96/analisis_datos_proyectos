import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import radians, cos, sin, asin, sqrt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Distancia Haversine para las coordenadas geograficas
# Calcula la distancia entre el dueño de la tarjeta y el comercio
def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radio de la tierra en km
    return c * r

# Cargar datos
df = pd.read_csv('3. credit_fraud/fraudTest.csv')

# --- 1. Extraemos variables importantes a partir del dataset ---

# Convertir fechas
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='mixed')
df['dob'] = pd.to_datetime(df['dob'], format='mixed')

# Calcular edad ya que es una variable demografica clave
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

# Extraer hora y dia
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()

# Función para calcular distancia Haversine para evaluar posibles por sus distancias geográficas
df['distance_km'] = haversine_vectorized(df['long'], df['lat'], df['merch_long'], df['merch_lat'])
print("Variables creadas: age, hour, day_of_week, distance_km")

# --- 2. Pruebas de normalidad  ---
# Usamos KS en lugar de Shapiro porque el dataset es probablemente grande (>5000 datos)
print("\n--- A. Prueba de Normalidad (Kolmogorov-Smirnov) ---")

numeric_vars = ['amt', 'distance_km', 'age', 'city_pop']
normality_results = {}
for col in numeric_vars:
    # Usamos muestra de 5000 para no saturar, pero suficiente para significancia
    stat, p = stats.kstest(df[col].dropna().sample(5000, random_state=42), 'norm')
    
    is_normal = p > 0.05
    normality_results[col] = "Normal" if is_normal else "No Normal"
    print(f"Variable '{col}': p-value={p:.4f} -> {normality_results[col]}")

# Conclusión automática del código
if any(res == "No Normal" for res in normality_results.values()):
    method_corr = "spearman"
    print("\n>>> CONCLUSIÓN: Se detectaron variables NO normales. Usaremos correlación de SPEARMAN.")
else:
    method_corr = "pearson"
    print("\n>>> CONCLUSIÓN: Todo es normal. Usaremos correlación de PEARSON.")

print(f"\n--- B: MATRIZ DE CORRELACIÓN númerica ({method_corr.upper()}) ---")

# ----------------- Correlacion de Spearman debido a variables no normales y por la cantidad de datos
cols_corr = ['amt', 'age', 'distance_km', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']
corr_matrix = df[cols_corr].corr(method=method_corr)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title(f'Matriz de Correlación ({method_corr.capitalize()})')
plt.savefig('correlation_matrix.png')
plt.close()

# ==============================================================================
# PRUEBAS DE HIPÓTESIS (Relación con el Fraude)
# Objetivo: Validar si la variable distingue entre Fraude (1) y No Fraude (0)
# ==============================================================================
print("\n--- PASO 3: PRUEBAS DE SIGNIFICANCIA (Variables vs Target) ---")

# A. VARIABLES NUMÉRICAS (Mann-Whitney U, debido a su distribucion no normal)
print(">>> A. Numéricas: Mann-Whitney U Test")
fraude = df[df['is_fraud'] == 1]
no_fraude = df[df['is_fraud'] == 0]

stats_results = []
for col in ['amt', 'distance_km', 'age', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']:
    stat, p = stats.mannwhitneyu(fraude[col], no_fraude[col])
    # Calculamos diferencia de medianas para ver el "tamaño del efecto" real
    diff_mediana = fraude[col].median() - no_fraude[col].median()
    stats_results.append({
        'Variable': col,
        'P-Value': p,
        'Es Significativo?': 'SÍ' if p < 0.05 else 'NO',
        'Diferencia Medianas': diff_mediana
    })

print(pd.DataFrame(stats_results))

# B. VARIABLES CATEGÓRICAS (Chi-Cuadrado)
print("\n>>> B. Categóricas: Chi-Cuadrado de Independencia")
# Probamos Gender y Category
for col in ['gender', 'category', 'state', 'job', 'day_of_week']:
    contingency = pd.crosstab(df[col], df['is_fraud'])
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"Variable '{col}': p-value={p:.4e} -> {'Significativo' if p < 0.05 else 'Independiente (Borrar)'}")

# =================== Eliminar variables en base a la importancia y análisis previo

df.drop(columns=["first","last","gender","street","job","dob","unix_time","city","state","sn","merchant","trans_num","cc_num","trans_date_trans_time"],inplace=True)
df['category'].unique()

# Codificar variable categóricas
le = LabelEncoder()
df["category"] =le.fit_transform(df["category"])
df["day_of_week"] =le.fit_transform(df["day_of_week"])

X = df.drop(columns=["is_fraud"])
Y = df["is_fraud"] 

# Estandarizar los valores numéricos
sc = StandardScaler()
sc.fit(X)
X = pd.DataFrame(sc.transform(X),columns=X.columns)

# ============================= Aplicar Under Sampling para balancear las clases
print("\n--- Logistic Regression con Under Sampling ---")
ru = RandomUnderSampler()
ru_x,ru_y =ru.fit_resample(X,Y)
print(ru_y.value_counts())

x_train,x_test,y_train,y_test = train_test_split(ru_x,ru_y,test_size= 0.2,random_state =42)
lr = LogisticRegression()
lr.fit(x_train,y_train)
cf = confusion_matrix(y_test,lr.predict(x_test))
print(cf)

sns.heatmap(cf,annot=True)
print("Precision:", precision_score(y_test,lr.predict(x_test))*100)
print("Recall:", recall_score(y_test,lr.predict(x_test))*100)
print("F1 Score:", f1_score(y_test,lr.predict(x_test))*100)

# ============================= Aplicar Over sampling para balancear las clases
print("\n--- Logistic Regression con Over Sampling (SMOTE) ---")
sm = SMOTE()
x_s,y_s = sm.fit_resample(X,Y)
x_train_s,x_test_s,y_train_s,y_test_s = train_test_split(x_s,y_s,test_size= 0.2,random_state =42)
lr =  LogisticRegression()
lr.fit(x_train_s,y_train_s)
cf_2 = confusion_matrix(y_test_s,lr.predict(x_test_s))
print(cf_2)

print(sns.heatmap(cf_2,annot=True))
print("Precision:", precision_score(y_test_s,lr.predict(x_test_s))*100)
print("Recall:", recall_score(y_test_s,lr.predict(x_test_s))*100)
print("F1 Score:", f1_score(y_test_s,lr.predict(x_test_s))*100)

# ============================= Estudios con Decision Tree Classifier

# Under sampling
print("\n--- Decision Tree Classifier con Under Sampling ---")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train,y_train)
cf_4 = confusion_matrix(y_test,dt.predict(x_test))
print(sns.heatmap(cf_4,annot=True))
print("Precision:", precision_score(y_test,dt.predict(x_test))*100)
print("Recall:", recall_score(y_test,dt.predict(x_test))*100)
print("F1 Score:", f1_score(y_test,dt.predict(x_test))*100)

# Over sampling
print("\n--- Decision Tree Classifier con Over Sampling ---")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train_s,y_train_s)
cf_3 = confusion_matrix(y_test_s,dt.predict(x_test_s))
print(sns.heatmap(cf_3,annot=True))
print("Precision:", precision_score(y_test_s,dt.predict(x_test_s))*100)
print("Recall:", recall_score(y_test_s,dt.predict(x_test_s))*100)
print("F1 Score:", f1_score(y_test_s,dt.predict(x_test_s))*100)

# ============================= Graficamos el arbol de decision logradio
#plt.figure(figsize=(20, 10))
# Visualizar el árbol entrenado con UnderSampling (es más fácil de leer)
#plot_tree(dt, 
#          feature_names=X.columns,       # Nombres de tus variables (amt, age, etc.)
#          class_names=['No Fraude', 'Fraude'], # Etiquetas (0, 1)
#          filled=True,                   # Colores (Azul=Fraude, Naranja=No Fraude usualmente)
#          rounded=True, 
#          fontsize=10)
#plt.title("Árbol de Decisión para Detección de Fraude (Reglas Aprendidas)")
#plt.show()

#  ============================= Comparacion con una neurona clasificadora usando tensorflow keras
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_s, 
          y_train_s, 
          epochs=10, 
          batch_size=32, 
          validation_split=0.2)

loss, accuracy = model.evaluate(x_test_s, y_test_s)

print(f"Neurona Clasificadora - Pérdida: {loss:.4f}, Precisión: {accuracy*100:.2f}%")
# Predicciones y matriz de confusión
y_pred_keras = (model.predict(x_test_s) > 0.5).astype("int32")
cf_keras = confusion_matrix(y_test_s, y_pred_keras)
print(sns.heatmap(cf_keras, annot=True))
print("Precision:", precision_score(y_test_s, y_pred_keras)*100)
print("Recall:", recall_score(y_test_s, y_pred_keras)*100)
print("F1 Score:", f1_score(y_test_s, y_pred_keras)*100)


