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
import kagglehub
from kagglehub import KaggleDatasetAdapter
from keras.optimizers import SGD, RMSprop, Adam, Lamb



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
#df = pd.read_csv("/kaggle/input/credit-card-fraud-data/fraudTest.csv")

# Cargar el dataset desde Kaggle usando kagglehub
file_path = "fraudTest.csv"
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "chetanmittal033/credit-card-fraud-data",
  file_path
)
print("Datos cargados. Dimensiones del dataset:", df.shape)

# --- Extraemos variables importantes a partir del dataset ---

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

# --- Pruebas de normalidad  ---
# Usamos KS en lugar de Shapiro es grande >5000 datos
print("\n--- Prueba de Normalidad (Kolmogorov-Smirnov) ---")

numeric_vars = ['amt', 'distance_km', 'age', 'city_pop']
normality_results = {}
for col in numeric_vars:
    # Usamos muestra de 5000 para no saturar, pero suficiente para significancia
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

# ----------------- Correlacion de Spearman debido a variables no normales y por la cantidad de datos
cols_corr = ['amt', 'age', 'distance_km', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']
corr_matrix = df[cols_corr].corr(method=method_corr)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title(f'Matriz de Correlación ({method_corr.capitalize()})')
plt.savefig('3. credit_fraud/correlation_matrix.png')
plt.close()

# ======================== Pruebas de hipotesis estadisticas para ver que variables son significativas
print("\n--- PRUEBAS DE SIGNIFICANCIA  ---")

# Variables numericas (Mann-Whitney U, debido a su distribucion no normal)
print("Numéricas: Mann-Whitney U Test")
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

# Variables categoricas (Chi-Cuadrado)
print("\nCategóricas: Chi-Cuadrado de Independencia")
# Probamos Gender y Category
for col in ['gender', 'category', 'state', 'job', 'day_of_week']:
    contingency = pd.crosstab(df[col], df['is_fraud'])
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"Variable '{col}': p-value={p:.4e} -> {'Significativo' if p < 0.05 else 'Independiente (Borrar)'}")

# =================== Eliminar variables en base a la importancia y análisis previo

df.drop(columns=["first","last","gender","street","dob","unix_time","city","state","sn","merchant","trans_num","cc_num","trans_date_trans_time", "lat", "merch_lat", "distance_km"],inplace=True)
df['category'].unique()

# Codificar variable categóricas
le = LabelEncoder()
cat_cols_toEncode = ["category", "day_of_week", "job"]  
for col in cat_cols_toEncode:
    if col in df.columns:   # If para evitar errores si la columna ya fue eliminada
        df[col] = le.fit_transform(df[col])

X = df.drop(columns=["is_fraud"])
Y = df["is_fraud"] 

# Estandarizar los valores numéricos
sc = StandardScaler()
sc.fit(X)
X = pd.DataFrame(sc.transform(X),columns=X.columns)

# ============================= Probar Logistic Regression con class_weight='balanced'

print("\n--- Logistic Regression con class_weight='balanced' ---")
lr_bal = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
x_train_bal, x_test_bal,y_train_bal,y_test_bal = train_test_split(X,Y,test_size= 0.2,random_state =42, stratify=Y)
lr_bal.fit(x_train_bal, y_train_bal)
y_pred_bal = lr_bal.predict(x_test_bal)

print(confusion_matrix(y_test_bal, y_pred_bal))
print("Precision:", precision_score(y_test_bal, y_pred_bal)*100)
print("Recall:", recall_score(y_test_bal, y_pred_bal)*100)
print("F1:", f1_score(y_test_bal, y_pred_bal)*100)

'''
# PROBAR DIFERENTES UMBRALES DE DECISION
y_proba = lr_bal.predict_proba(x_test_bal)[:,1]

for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y_pred_t = (y_proba >= t).astype(int)
    print(f"\nThreshold = {t}")
    print("Precision:", precision_score(y_test_bal, y_pred_t)*100)
    print("Recall:", recall_score(y_test_bal, y_pred_t)*100)
    print("F1:", f1_score(y_test_bal, y_pred_t)*100)
'''
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
# AQUI, es importante que usar el smote SOLO en los datos de entrenamiento para evitar data leakage 
print("\n--- Logistic Regression con Over Sampling (SMOTE) ---")

x_train_over, x_test_over,y_train_over,y_test_over = train_test_split(X,Y,test_size= 0.2,random_state =42, stratify=Y)
sm = SMOTE()
x_s_over,y_s_over = sm.fit_resample(x_train_over,y_train_over)

lr =  LogisticRegression(
    class_weight="balanced",
)
lr.fit(x_s_over,y_s_over)
cf_2 = confusion_matrix(y_test_over,lr.predict(x_test_over))
print(cf_2)

print(sns.heatmap(cf_2,annot=True))
print("Precision:", precision_score(y_test_over,lr.predict(x_test_over))*100)
print("Recall:", recall_score(y_test_over,lr.predict(x_test_over))*100)
print("F1 Score:", f1_score(y_test_over,lr.predict(x_test_over))*100)

# ============================= Estudios con Decision Tree Classifier
# =========== Con class_weight='balanced'
print("\n--- Decision Tree Classifier con class_weight='balanced' ---")
dt_bal = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

dt_bal.fit(x_train, y_train)
y_pred_dt = dt_bal.predict(x_test)

print(confusion_matrix(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt)*100)
print("Recall:", recall_score(y_test, y_pred_dt)*100)
print("F1:", f1_score(y_test, y_pred_dt)*100)

# =========== Under sampling
print("\n--- Decision Tree Classifier con Under Sampling ---")
dt = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

dt.fit(x_train,y_train)
cf_4 = confusion_matrix(y_test,dt.predict(x_test))
print(sns.heatmap(cf_4,annot=True))
print("Precision:", precision_score(y_test,dt.predict(x_test))*100)
print("Recall:", recall_score(y_test,dt.predict(x_test))*100)
print("F1 Score:", f1_score(y_test,dt.predict(x_test))*100)

# =========== Over sampling
print("\n--- Decision Tree Classifier con Over Sampling ---")
dt = DecisionTreeClassifier(

    class_weight="balanced",
    random_state=42
)
dt.fit(x_s_over,y_s_over)
cf_3 = confusion_matrix(y_test_over,dt.predict(x_test_over))
print(sns.heatmap(cf_3,annot=True))
print("Precision:", precision_score(y_test_over,dt.predict(x_test_over))*100)
print("Recall:", recall_score(y_test_over,dt.predict(x_test_over))*100)
print("F1 Score:", f1_score(y_test_over,dt.predict(x_test_over))*100)

'''
# PARA VISUALIZAR EL ARBOL DE DECISION
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
'''

'''
# Usando over sampling
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_s_over, 
          y_s_over, 
          epochs=10, 
          batch_size=32,
          validation_split=0.2)

loss, accuracy = model.evaluate(x_test_over, y_test_over)

print(f"Neurona Clasificadora OVER - Pérdida: {loss:.4f}, Precisión: {accuracy*100:.2f}%")
# Predicciones y matriz de confusión
y_pred_keras = (model.predict(x_test_over) > 0.5).astype("int32")
cf_keras = confusion_matrix(y_test_over, y_pred_keras)
print(sns.heatmap(cf_keras, annot=True))
print("Precision:", precision_score(y_test_over, y_pred_keras)*100)
print("Recall:", recall_score(y_test_over, y_pred_keras)*100)
print("F1 Score:", f1_score(y_test_over, y_pred_keras)*100)
'''
#  ============================= Comparacion con una neurona clasificadora usando tensorflow keras

# Usando Under Sampling
model_under = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

algoritmos_optimizacion = {
        'SGD': SGD(learning_rate=0.001),
        'RMSprop': RMSprop(learning_rate=0.001),
        'Adam': Adam(learning_rate=0.001),
        'Lamb': Lamb(learning_rate=0.001)  
        }

resultados = []

for nombre_opt, optimizador in algoritmos_optimizacion.items():
    print(f'\nEntrenando con optimizador: {nombre_opt}')

    model_under.compile(optimizer=optimizador,
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model_under.fit(x_train,
                    y_train,
            epochs=100, 
            batch_size=32,
            validation_split=0.2,
            verbose=0
            )

    loss, accuracy = model_under.evaluate(x_test, y_test)
    print(f"Neurona Clasificadora UNDER - Pérdida: {loss:.4f}, Precisión: {accuracy*100:.2f}%")
    
    # Predicciones y matriz de confusión
    y_pred_keras = (model_under.predict(x_test) > 0.5).astype("int32")
    #cf_keras = confusion_matrix(y_test, y_pred_keras)
    #print(sns.heatmap(cf_keras, annot=True))
    #print("Precision:", precision_score(y_test, y_pred_keras)*100)
    #print("Recall:", recall_score(y_test, y_pred_keras)*100)
    #print("F1 Score:", f1_score(y_test, y_pred_keras)*100)

    # Almacenar resultados para imprimir la tabla comparativa final
    resultados.append({
        'Optimizador': nombre_opt,
        'Pérdida': loss,
        'Precisión': accuracy * 100,
        'Precision': precision_score(y_test, y_pred_keras)*100,
        'Recall': recall_score(y_test, y_pred_keras)*100,
        'F1 Score': f1_score(y_test, y_pred_keras)*100
    })

# Imprimir tabla comparativa final
print("\n--- Resultados Comparativos de Optimizadores ---")
print(pd.DataFrame(resultados))
