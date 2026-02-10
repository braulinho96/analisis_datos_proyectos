import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import kagglehub
import os
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score,  roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop, Adam, Lamb
from tensorflow.keras import Sequential, layers, metrics, callbacks
from kagglehub import KaggleDatasetAdapter
from torch import neg

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()

# Ordenamos por tarjeta y fecha para calcular características temporales
df = df.sort_values(by=['cc_num', 'trans_date_trans_time'])

# Features de Velocidad
df_temp = df.set_index('trans_date_trans_time')
df['trans_count_24h'] = df_temp.groupby('cc_num')['amt'].rolling('24h').count().values
df['avg_amt_last_5'] = df.groupby('cc_num')['amt'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['diff_from_avg'] = df['amt'] - df['avg_amt_last_5']


# Codificar variable categóricas
le = LabelEncoder()
cat_cols_toEncode = ["category",'day_of_week']  
for col in cat_cols_toEncode:
    if col in df.columns:   
        df[col] = le.fit_transform(df[col])
        
df = df.sort_values(by=['trans_date_trans_time'])

# =================== Eliminar variables ===================
cols_to_drop = ["first","last","gender","street","dob","unix_time","city","state",
                "sn","merchant","trans_num","cc_num","trans_date_trans_time", 
                "long", "merch_long", 'zip', 'city_pop', 'job']

df.drop(columns=cols_to_drop, inplace=True)

# Definimos DE X e Y
X = df.drop(columns=["is_fraud"])
Y = df["is_fraud"] 

# ================= Split Temporal =================
'''
# Si balanceamos con stratify, el modelo no aprende a detectar fraudes porque el dataset es muy pequeño y el 99% de las transacciones son buenas.
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.2, 
    random_state=42, 
    stratify=Y)

scaler_full = StandardScaler()
X_train_full_sc = scaler_full.fit_transform(X_train)
X_test_full_sc = scaler_full.transform(X_test)
'''

# Definir el punto de corte usar el último 20% del tiempo para test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]

# Arboles de deicision y random forest no necesitan escalado, pero para la red neuronal si es recomendable
# Usamos RobustScaler para reducir el impacto de outliers, que son comunes en datos de fraude
scaler_full = RobustScaler()
X_train_full_sc = scaler_full.fit_transform(X_train)
X_test_full_sc = scaler_full.transform(X_test)

# =========== DECISION TREE CLASSIFIER CON CLASS WEIGHT BALANCED ===========
print("\n--- Decision Tree Classifier con class_weight='balanced' ---")
dt_bal = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt_bal.fit(X_train_full_sc, y_train)
y_pred_dt = dt_bal.predict(X_test_full_sc)

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

rf_model.fit(X_train_full_sc, y_train)

# Iteramos sobre varios thresholds y evaluamos
for threshold in [0.3, 0.5, 0.7]:
    print(f"\n--- Threshold: {threshold} ---")
    y_proba_rf = rf_model.predict_proba(X_test_full_sc)[:, 1]
    y_pred_rf_thresh = (y_proba_rf > threshold).astype(int)
    print(confusion_matrix(y_test, y_pred_rf_thresh))
    print(classification_report(y_test, y_pred_rf_thresh))
    print("Precision:", precision_score(y_test, y_pred_rf_thresh)*100)
    print("Recall:", recall_score(y_test, y_pred_rf_thresh)*100)
    print("F1 Score:", f1_score(y_test, y_pred_rf_thresh)*100)

# Curva ROC
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba_rf)
roc_auc = roc_auc_score(y_test, y_proba_rf)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.savefig("3. credit_fraud/roc_curve_rf.png")
plt.show()
plt.close()

# Curva Precision-Recall
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba_rf)
ap_score = average_precision_score(y_test, y_proba_rf)
plt.figure()
plt.plot(recall, precision, label=f"AP = {ap_score:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve - Random Forest")
plt.legend()
plt.savefig("3. credit_fraud/precision_recall_curve_rf.png")
plt.show()
plt.close()

# Importancia de características mediante Permutation Importance
result = permutation_importance(
    rf_model,
    X_test_full_sc,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="average_precision"
)

importances = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(
    data=importances.head(15),
    x="importance_mean",
    y="feature",
    orient="h"
)
plt.title("Permutation Feature Importance (Random Forest)")
plt.xlabel("Decrease in F1-score")
plt.ylabel("Feature")
plt.savefig("3. credit_fraud/permutation_importance_rf.png")
plt.show()

# =================== Deep Neural Network ===================
print("\n--- Deep Neural Network ---")

# Debido al desbalance de clases, hay que ajustar el umbral de decisión
negativo, positivo = np.bincount(y_train)
initial_bias = np.log([positivo / negativo])
# Calculamos class weights para el entrenamiento
class_weights = {0: (1/negativo)*(len(y_train)/2.0), 1: (1/positivo)*(len(y_train)/2.0)}
output_bias = tf.keras.initializers.Constant(initial_bias)

model_dnn = keras.Sequential([
    layers.Input(shape=(X_train_full_sc.shape[1],)),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    
    # Corregimos el sesgo inicial para que el modelo empiece con una predicción más informada sobre la clase minoritaria
    # Es decir, le damos una probabilidad inicial basada en la distribución de clases
    #layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    
    # Resulta que con el bias inicial el modelo no aprende a detectar fraudes, así que lo dejamos sin bias y con class weights
    layers.Dense(1, activation='sigmoid') 
])

model_dnn.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        metrics.Recall(name="recall"),
        metrics.AUC(name="pr_auc", curve="PR")
    ]
)

# Early stopping basado en PR AUC, para detener el entrenamiento cuando la métrica deje de mejorar
early_stop = callbacks.EarlyStopping(
    monitor='val_pr_auc', 
    mode='max', 
    patience=40, 
    restore_best_weights=True
)

model_dnn.fit(
    X_train_full_sc,
    y_train,
    epochs=1000,
    batch_size=2048, # Usamos un batch grande para estabilizar el entrenamiento con datos desbalanceados
    validation_split=0.2,
    #class_weight=class_weights,
    verbose=1,
    callbacks=[early_stop]
)

y_proba_full = model_dnn.predict(X_test_full_sc).ravel()

for threshold in [0.3, 0.5, 0.7]:
    print(f"--- Threshold: {threshold} ---")
    y_pred_thresh = (y_proba_full > threshold).astype(int)
    print(confusion_matrix(y_test, y_pred_thresh))
    print(classification_report(y_test, y_pred_thresh))
    print(f"Precision: {precision_score(y_test, y_pred_thresh)*100:.2f}%")
    print(f"Recall:    {recall_score(y_test, y_pred_thresh)*100:.2f}%")
    print(f"F1 Score:  {f1_score(y_test, y_pred_thresh)*100:.2f}%")

# Curva ROC
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba_full)
roc_auc = roc_auc_score(y_test, y_proba_full)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Deep Neural Network")
plt.legend()
plt.savefig("3. credit_fraud/roc_curve_dnn.png")
plt.show()
plt.close()

# Curva Precision-Recall
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba_full)
ap_score = average_precision_score(y_test, y_proba_full)
plt.figure()
plt.plot(recall, precision, label=f"AP = {ap_score:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve - Deep Neural Network")
plt.legend()
plt.savefig("3. credit_fraud/precision_recall_curve_dnn.png")
plt.show()
plt.close()

'''
def f1_scorer(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return f1_score(y_true, y_pred)

def permutation_importance_nn(model, X, y, threshold=0.5, n_repeats=5):
    baseline = f1_scorer(y, model.predict(X).ravel(), threshold)
    importances = []

    for col in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, col])
            score = f1_scorer(y, model.predict(X_permuted).ravel(), threshold)
            scores.append(baseline - score)
        importances.append(np.mean(scores))

    return np.array(importances)

# Evaluamos la 
nn_importances = permutation_importance_nn(
    model_dnn,
    X_test_full_sc,
    y_test,
    threshold=0.5,
    n_repeats=2
)

nn_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": nn_importances
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(
    data=nn_importance_df.head(15),
    x="importance",
    y="feature",
    orient="h"
)
plt.title("Permutation Feature Importance (Neural Network)")
plt.xlabel("Decrease in F1-score")
plt.ylabel("Feature")
plt.savefig("3. credit_fraud/permutation_importance_dnn.png")
plt.show()
'''

