import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import StandardScaler
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
# Nota: Lamb no suele estar en keras.optimizers estándar, verifica si necesitas tensorflow_addons
from tensorflow.keras.optimizers import RMSprop, Adam, Lamb
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.utils import class_weight

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Distancia Haversine para las coordenadas geograficas
def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radio de la tierra en km
    return c * r

# Cargar el dataset desde Kaggle usando kagglehub
file_path = "fraudTest.csv"
try:
    df = kagglehub.dataset_load(
      KaggleDatasetAdapter.PANDAS,
      "chetanmittal033/credit-card-fraud-data",
      file_path
    )
    print("Datos cargados. Dimensiones del dataset:", df.shape)
except Exception as e:
    print("Error cargando kagglehub, asegúrate de tener los datos:", e)
    # df = pd.read_csv("fraudTest.csv") # Descomentar si usas local

# --- Extraemos variables importantes a partir del dataset ---

# Convertir fechas
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='mixed')
df['dob'] = pd.to_datetime(df['dob'], format='mixed')

# Calcular edad 
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

# Extraer hora y dia
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()

# Función para calcular distancia Haversine 
df['distance_km'] = haversine_vectorized(df['long'], df['lat'], df['merch_long'], df['merch_lat'])
print("Variables creadas: age, hour, day_of_week, distance_km")

# --- Pruebas de normalidad ---
print("\n--- Prueba de Normalidad (Kolmogorov-Smirnov) ---")

numeric_vars = ['amt', 'distance_km', 'age', 'city_pop']
normality_results = {}
for col in numeric_vars:
    # Usamos muestra de 5000 para no saturar
    stat, p = stats.kstest(df[col].dropna().sample(5000, random_state=42), 'norm')
    is_normal = p > 0.05
    normality_results[col] = "Normal" if is_normal else "No Normal"
    print(f"Variable '{col}': p-value={p:.4f} -> {normality_results[col]}")

# Asignacion segun normalidad
if any(res == "No Normal" for res in normality_results.values()):
    method_corr = "spearman"
    print("\n>>> CONCLUSIÓN: Se detectaron variables NO normales. Usaremos correlación de SPEARMAN.")
else:
    method_corr = "pearson"
    print("\n>>> CONCLUSIÓN: Todo es normal. Usaremos correlación de PEARSON.")

print(f"\n--- Matriz de correlacion ({method_corr.upper()}) ---")

# ----------------- Correlacion 
cols_corr = ['amt', 'age', 'distance_km', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']
corr_matrix = df[cols_corr].corr(method=method_corr)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title(f'Matriz de Correlación ({method_corr.capitalize()})')
# plt.savefig('3. credit_fraud/correlation_matrix.png')
plt.close()

# ======================== Pruebas de hipotesis estadisticas 
print("\n--- PRUEBAS DE SIGNIFICANCIA ---")

# Variables numericas (Mann-Whitney U)
print("Numéricas: Mann-Whitney U Test")
fraude = df[df['is_fraud'] == 1]
no_fraude = df[df['is_fraud'] == 0]

stats_results = []
for col in ['amt', 'distance_km', 'age', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']:
    stat, p = stats.mannwhitneyu(fraude[col], no_fraude[col])
    diff_mediana = fraude[col].median() - no_fraude[col].median()
    stats_results.append({
        'Variable': col,
        'P-Value': p,
        'Es Significativo?': 'SÍ' if p < 0.05 else 'NO',
        'Diferencia Medianas': diff_mediana
    })

print(pd.DataFrame(stats_results))

# Variables categoricas (Chi-Cuadrado)
print("\nCategóricas: Chi-Cuadrado de Independencia")
for col in ['gender', 'category', 'state', 'job', 'day_of_week']:
    contingency = pd.crosstab(df[col], df['is_fraud'])
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"Variable '{col}': p-value={p:.4e} -> {'Significativo' if p < 0.05 else 'Independiente (Borrar)'}")

# =================== Eliminar variables ===================

df.drop(columns=["first","last","gender","street","dob","unix_time","city","state","sn","merchant","trans_num","cc_num","trans_date_trans_time", "lat", "merch_lat", "distance_km"],inplace=True)

# Codificar variable categóricas
le = LabelEncoder()
cat_cols_toEncode = ["category", "day_of_week", "job"]  
for col in cat_cols_toEncode:
    if col in df.columns:   
        df[col] = le.fit_transform(df[col])

# DEFINICIÓN DE X e Y
X = df.drop(columns=["is_fraud"])
Y = df["is_fraud"] 

# ==============================================================================
# SECCIÓN CRÍTICA DE PRE-PROCESAMIENTO (CORREGIDA)
# ==============================================================================
# Hacemos el Split y el Scaling UNA sola vez para asegurar consistencia
# y evitar Data Leakage en todos los modelos subsiguientes.

print("\n--- PREPARANDO DATOS (SPLIT & SCALE GLOBAL) ---")

# 1. SPLIT: Separamos Train (80%) y Test (20%) - EL TEST ES SAGRADO
x_train_global, x_test_global, y_train_global, y_test_global = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# 2. SCALING: Ajustamos (fit) solo con Train, transformamos ambos
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train_global)
x_test_sc = sc.transform(x_test_global) # Este se usará para evaluar TODOS los modelos

# 3. PREPARAR DATASETS PARA TÉCNICAS DE SAMPLING (Solo modificamos el Train)

# A. Under Sampling (Para usar luego)
us = RandomUnderSampler(random_state=42)
x_train_us, y_train_us = us.fit_resample(x_train_sc, y_train_global)
print(f"Dimensiones UnderSampling Train: {x_train_us.shape}")

# B. Over Sampling SMOTE (Para usar luego)
sm = SMOTE(random_state=42)
x_train_sm, y_train_sm = sm.fit_resample(x_train_sc, y_train_global)
print(f"Dimensiones SMOTE Train: {x_train_sm.shape}")
print("-" * 50)


# ============================= Logistic Regression con class_weight='balanced'
print("\n--- Logistic Regression con class_weight='balanced' ---")

lr_bal = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr_bal.fit(x_train_sc, y_train_global) # Entrenamos con datos escalados originales
y_pred_bal = lr_bal.predict(x_test_sc) # Evaluamos con test escalado original

print(confusion_matrix(y_test_global, y_pred_bal))
print("Precision:", precision_score(y_test_global, y_pred_bal)*100)
print("Recall:", recall_score(y_test_global, y_pred_bal)*100)
print("F1:", f1_score(y_test_global, y_pred_bal)*100)

'''
# PROBAR DIFERENTES UMBRALES DE DECISION
y_proba = lr_bal.predict_proba(x_test_sc)[:,1] # Corregido para usar el sc

for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y_pred_t = (y_proba >= t).astype(int)
    print(f"\nThreshold = {t}")
    print("Precision:", precision_score(y_test_global, y_pred_t)*100)
    print("Recall:", recall_score(y_test_global, y_pred_t)*100)
    print("F1:", f1_score(y_test_global, y_pred_t)*100)
'''

# ============================= Logistic Regression con Under Sampling
print("\n--- Logistic Regression con Under Sampling ---")

# Usamos los datos ya preparados arriba (x_train_us)
lr_us = LogisticRegression(max_iter=1000)
lr_us.fit(x_train_us, y_train_us)

y_pred_us = lr_us.predict(x_test_sc) # Evaluamos en el test real
cf = confusion_matrix(y_test_global, y_pred_us)

print(cf)
# sns.heatmap(cf, annot=True, fmt='d') # Descomenta si quieres ver el gráfico
# plt.show()

print("Precision:", precision_score(y_test_global, y_pred_us)*100)
print("Recall:", recall_score(y_test_global, y_pred_us)*100)
print("F1 Score:", f1_score(y_test_global, y_pred_us)*100)

# ============================= Logistic Regression con Over Sampling (SMOTE)
print("\n--- Logistic Regression con Over Sampling (SMOTE) ---")

# Usamos los datos ya preparados arriba (x_train_sm)
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(x_train_sm, y_train_sm)

y_pred_smote = lr_smote.predict(x_test_sc) # Evaluamos en el test real
cf_2 = confusion_matrix(y_test_global, y_pred_smote)

print(cf_2)
# sns.heatmap(cf_2, annot=True, fmt='d', cmap='Greens')
# plt.show()

print("Precision:", precision_score(y_test_global, y_pred_smote)*100)
print("Recall:", recall_score(y_test_global, y_pred_smote)*100)
print("F1 Score:", f1_score(y_test_global, y_pred_smote)*100)

# ============================= Estudios con Decision Tree Classifier

# =========== Con class_weight='balanced'
print("\n--- Decision Tree Classifier con class_weight='balanced' ---")
dt_bal = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt_bal.fit(x_train_sc, y_train_global)
y_pred_dt = dt_bal.predict(x_test_sc)

print(confusion_matrix(y_test_global, y_pred_dt))
print("Precision:", precision_score(y_test_global, y_pred_dt)*100)
print("Recall:", recall_score(y_test_global, y_pred_dt)*100)
print("F1:", f1_score(y_test_global, y_pred_dt)*100)

# =========== Under sampling
print("\n--- Decision Tree Classifier con Under Sampling ---")
dt_u = DecisionTreeClassifier(random_state=42) 
dt_u.fit(x_train_us, y_train_us)
y_pred_dt_u = dt_u.predict(x_test_sc)

print(confusion_matrix(y_test_global, y_pred_dt_u))
print("Precision:", precision_score(y_test_global, y_pred_dt_u)*100)
print("Recall:", recall_score(y_test_global, y_pred_dt_u)*100)
print("F1 Score:", f1_score(y_test_global, y_pred_dt_u)*100)

# =========== Over sampling (SMOTE)
print("\n--- Decision Tree Classifier con Over Sampling ---")
dt_s = DecisionTreeClassifier(random_state=42) 
dt_s.fit(x_train_sm, y_train_sm)
y_pred_dt_s = dt_s.predict(x_test_sc)

print(confusion_matrix(y_test_global, y_pred_dt_s))
print("Precision:", precision_score(y_test_global, y_pred_dt_s)*100)
print("Recall:", recall_score(y_test_global, y_pred_dt_s)*100)
print("F1 Score:", f1_score(y_test_global, y_pred_dt_s)*100)

#  ============================= Comparacion con una neurona clasificadora 

# Usando Under Sampling (Tal como indicaba el comentario original)
# IMPORTANTE: Definimos los optimizadores disponibles
algoritmos_optimizacion = {
        'RMSprop': RMSprop(learning_rate=0.001),
        'Adam': Adam(learning_rate=0.001),
        'Lamb': Lamb(learning_rate=0.001) 
        }

resultados_keras = []

print("\n--- Entrenando Redes Neuronales (Under Sampling) ---")
for nombre_opt, optimizador in algoritmos_optimizacion.items():
    print(f'\nEntrenando con optimizador: {nombre_opt}')

    model_under = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(x_train_us.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model_under.compile(optimizer=optimizador,
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model_under.fit(x_train_us,
                    y_train_us,
            epochs=50, 
            batch_size=32,
            validation_split=0.2,
            verbose=0
            )

    loss, accuracy = model_under.evaluate(x_test_sc, y_test_global, verbose=0)
    print(f"Resultado {nombre_opt} - Pérdida: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")

    # Predicciones
    y_pred_keras = (model_under.predict(x_test_sc) > 0.5).astype("int32")

    resultados_keras.append({
        'Optimizador': nombre_opt,
        'Pérdida': loss,
        'Precisión (Acc)': accuracy * 100,
        'Precision': precision_score(y_test_global, y_pred_keras)*100,
        'Recall': recall_score(y_test_global, y_pred_keras)*100,
        'F1 Score': f1_score(y_test_global, y_pred_keras)*100
    })

# Imprimir tabla comparativa final
print("\n--- Resultados Comparativos de Optimizadores (Keras) ---")
print(pd.DataFrame(resultados_keras))



# ================== RANDOM FOREST CLASSIFIER CON CLASS WEIGHT BALANCED
print("\n--- Random Forest con Class Weight Balanced ---")

# Usamos los datos globales (x_train_sc) que ya están escalados y divididos correctamente
rf_model = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    random_state=42,
    max_depth=10, 
    n_jobs=-1
)
rf_model.fit(x_train_sc, y_train_global)
y_pred_rf = rf_model.predict(x_test_sc)

print(confusion_matrix(y_test_global, y_pred_rf))
print(classification_report(y_test_global, y_pred_rf))
print("Precision:", precision_score(y_test_global, y_pred_rf)*100)
print("Recall:", recall_score(y_test_global, y_pred_rf)*100)
print("F1 Score:", f1_score(y_test_global, y_pred_rf)*100)


print("\n--- Red Neuronal con Class Weights ---")
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_global),
    y=y_train_global
)
dict_weights = {0: weights[0], 1: weights[1]}

# 2. Definir una arquitectura un poco más robusta
model_weighted = keras.Sequential([
    layers.Input(shape=(x_train_sc.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(), # Estabiliza el aprendizaje
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_weighted.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# 3. Entrenar con el dataset COMPLETO y los pesos
model_weighted.fit(
    x_train_sc, y_train_global,
    epochs=50,
    batch_size=32, # Batch grande para que siempre haya algún fraude en cada paso
    class_weight=dict_weights,
    validation_split=0.1,
    verbose=0
)

# 4. Evaluación
y_pred_nn = (model_weighted.predict(x_test_sc) > 0.5).astype(int)
print(classification_report(y_test_global, y_pred_nn))
print("Precision:", precision_score(y_test_global, y_pred_nn)*100)
print("Recall:", recall_score(y_test_global, y_pred_nn)*100)
print("F1 Score:", f1_score(y_test_global, y_pred_nn)*100)


