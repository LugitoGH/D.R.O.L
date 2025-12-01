import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Carregar dados
dados = pd.read_csv("gestos_libras.csv", header=None)

# Limpar rótulos
dados[0] = dados[0].astype(str).str.strip()
dados = dados[dados[0] != ""]
dados = dados[dados[0].notna()]

# Separar X e y
X = dados.iloc[:, 1:]
y_raw = dados.iloc[:, 0]

# Transformar rótulos para números
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Treinar modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Acurácia
print(f"Acurácia: {modelo.score(X_test, y_test)*100:.2f}%")

# Salvar modelo e encoder
joblib.dump(modelo, "modelo_libras.pkl")
joblib.dump(le, "label_encoder.pkl")