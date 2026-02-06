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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder  
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam, Lamb
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.utils import class_weight

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

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

# =================== Eliminar variables ===================
df.drop(columns=["first","last","gender","street","dob","unix_time","city","state","sn","merchant","trans_num","cc_num","trans_date_trans_time", "lat", "merch_lat"],inplace=True)

# Codificar variable categóricas
le = LabelEncoder()
cat_cols_toEncode = ["category", "day_of_week", "job"]  
for col in cat_cols_toEncode:
    if col in df.columns:   
        df[col] = le.fit_transform(df[col])

# DEFINICIÓN DE X e Y
X = df.drop(columns=["is_fraud"])
Y = df["is_fraud"] 

# ================= SPLIT GLOBAL =================
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# =================================================
# MODELO A: CON UNDER SAMPLING
# =================================================
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

scaler_under = StandardScaler()
X_train_under_sc = scaler_under.fit_transform(X_train_under)
X_test_under_sc = scaler_under.transform(X_test)

# =================================================
# MODELO B: SIN UNDER SAMPLING
# =================================================
scaler_full = StandardScaler()
X_train_full_sc = scaler_full.fit_transform(X_train)
X_test_full_sc = scaler_full.transform(X_test)



# =================== Logistic Regression ===================
print("\n--- Logistic Regression usando pipeline' ---")
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        #class_weight={0:1, 1:10} # Ajustar manualmente los pesos
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred)*100)
print("Recall:", recall_score(y_test, y_pred)*100)
print("F1 Score:", f1_score(y_test, y_pred)*100)

# =========== DECISION TREE CLASSIFIER CON CLASS WEIGHT BALANCED ===========
print("\n--- Decision Tree Classifier con class_weight='balanced' ---")
dt_bal = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt_bal.fit(X_train, y_train)
y_pred_dt = dt_bal.predict(X_test)

print(confusion_matrix(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt)*100)
print("Recall:", recall_score(y_test, y_pred_dt)*100)
print("F1:", f1_score(y_test, y_pred_dt)*100)

# =================== Random Forest Classifier ===================
print("\n--- Random Forest Classifier con class_weight='balanced' ---")
rf_model = RandomForestClassifier( 
    n_estimators=100, 
    class_weight='balanced', 
    random_state=42, 
    n_jobs=-1 )

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf)*100)
print("Recall:", recall_score(y_test, y_pred_rf)*100)
print("F1 Score:", f1_score(y_test, y_pred_rf)*100)

# =================== Deep Neural Network  ===================
print("\n--- Deep Neural Network con under sampling ---")

print("\n--- Deep Neural Network con UnderSampling ---")

model_under = keras.Sequential([
    keras.Input(shape=(X_train_under_sc.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_under.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="pr_auc", curve="PR")
    ]
)

model_under.fit(
    X_train_under_sc,
    y_train_under,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

y_proba_under = model_under.predict(X_test_under_sc).ravel()
y_pred_under = (y_proba_under > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred_under))
print(classification_report(y_test, y_pred_under))
print("Precision:", precision_score(y_test, y_pred_under)*100)
print("Recall:", recall_score(y_test, y_pred_under)*100)
print("F1 Score:", f1_score(y_test, y_pred_under)*100)


# =================== Deep Neural Network SIN UnderSampling ===================
print("\n--- Deep Neural Network SIN UnderSampling ---")

model_full = keras.Sequential([
    keras.Input(shape=(X_train_full_sc.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_full.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="pr_auc", curve="PR")
    ]
)

model_full.fit(
    X_train_full_sc,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

y_proba_full = model_full.predict(X_test_full_sc).ravel()
y_pred_full = (y_proba_full > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred_full))
print(classification_report(y_test, y_pred_full))

print("Precision:", precision_score(y_test, y_pred_full)*100)
print("Recall:", recall_score(y_test, y_pred_full)*100)
print("F1 Score:", f1_score(y_test, y_pred_full)*100)