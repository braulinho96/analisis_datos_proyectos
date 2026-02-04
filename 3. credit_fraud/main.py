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
from sklearn.tree import DecisionTreeClassifier

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

def cramers_v(confusion_matrix):
    """Calcula la fuerza de asociación (0 a 1)"""
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

# Cargar datos
df = pd.read_csv('credit_fraud/fraudTest.csv')

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

# Incluimos coordenadas para ver su redundancia con distance_km
cols_corr = ['amt', 'age', 'distance_km', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']
corr_matrix = df[cols_corr].corr(method=method_corr)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title(f'Matriz de Correlación ({method_corr.capitalize()})')
plt.show()



# --- 3. VARIABLES NUMÉRICAS VS FRAUDE (Mann-Whitney U + Rank Biserial) ---
print("\n--- B. Comparación Numérica (Fraude vs No Fraude) ---")
numeric_cols = ['amt', 'distance_km', 'age', 'city_pop']
fraude = df[df['is_fraud'] == 1]
no_fraude = df[df['is_fraud'] == 0]

results_num = []

for col in numeric_cols:
    # Mann-Whitney U (Prueba de mediana para datos no paramétricos)
    stat, p = stats.mannwhitneyu(fraude[col], no_fraude[col], alternative='two-sided')
    
    # Calculamos la correlación Rank-Biserial (Tamaño del efecto)
    # Fórmula simplificada aproximada para grandes volúmenes o usar bibliotecas externas
    # Aquí reportamos la diferencia de medianas que es más interpretable
    median_diff = fraude[col].median() - no_fraude[col].median()
    
    results_num.append({
        'Variable': col,
        'Median_Fraud': fraude[col].median(),
        'Median_Legit': no_fraude[col].median(),
        'P-Value': p,
        'Significant': p < 0.05
    })

df_res_num = pd.DataFrame(results_num)
print(df_res_num)

# --- 4. VARIABLES CATEGÓRICAS VS FRAUDE (Chi-Cuadrado + V de Cramér) ---
print("\n--- C. Asociación Categórica (Chi-Cuadrado + V de Cramér) ---")
cat_cols = ['category', 'gender', 'day_of_week', 'state'] 
results_cat = []

for col in cat_cols:
    contingency = pd.crosstab(df[col], df['is_fraud'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    strength = cramers_v(contingency)
    results_cat.append({
        'Variable': col,
        'Chi2': chi2,
        'P-Value': p,
        'Cramers_V': strength,
        'Significant': p < 0.05
    })

df_res_cat = pd.DataFrame(results_cat)
print(df_res_cat)

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

# ============================= Aplicar Submuestreo Aleatorio para balancear las clases
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

# ============================= Aplicar Submuestreo Aleatorio para balancear las clases
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
dt = DecisionTreeClassifier()
dt.fit(x_train_s,y_train_s)
cf_3 = confusion_matrix(y_test_s,dt.predict(x_test_s))
print(sns.heatmap(cf_3,annot=True))
print("Precision:", precision_score(y_test_s,dt.predict(x_test_s))*100)
print("Recall:", recall_score(y_test_s,dt.predict(x_test_s))*100)
print("F1 Score:", f1_score(y_test_s,dt.predict(x_test_s))*100)

# Over sampling
print("\n--- Decision Tree Classifier con Over Sampling ---")
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
cf_4 = confusion_matrix(y_test,dt.predict(x_test))
print(sns.heatmap(cf_4,annot=True))
print("Precision:", precision_score(y_test,dt.predict(x_test))*100)
print("Recall:", recall_score(y_test,dt.predict(x_test))*100)
print("F1 Score:", f1_score(y_test,dt.predict(x_test))*100)


