import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

output_dir = 'features'
X = np.load(os.path.join(output_dir, 'X.npy'))
y = np.load(os.path.join(output_dir, 'y.npy'))

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  


# treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y_encoded, test_size=0.2, random_state=99, stratify=y)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_encoded, cmap='viridis', alpha=0.5)
plt.title("Visualização das Características")
plt.show()
'''
# knn
print("KNN")

knn = KNeighborsClassifier()

params = {
     "knn__n_neighbors": list(range(1, 30)), 
    "knn__metric": ['euclidean', 'manhattan', 'minkowski', 'cosine'],  
    "knn__weights": ['uniform', 'distance']  
}

pipe = Pipeline([
    ("scaler", StandardScaler()), # normalização
    ("knn", knn)
])

#validação cruzada
print("fazendo validação knn")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelo = GridSearchCV(pipe, params, cv=cv_strategy, n_jobs=-1)
modelo.fit(X_treino, y_treino)

melhor_param = modelo.best_params_
print("Melhores parâmetros: ", melhor_param)

acuracia = accuracy_score(y_teste, modelo.predict(X_teste)) 
print(f"Acurácia: {acuracia}")

print(classification_report(y_teste, modelo.predict(X_teste)))

ConfusionMatrixDisplay.from_estimator(modelo, X_teste, y_teste)
plt.show()
'''

# svm
print("SVM")

svm = SVC(kernel='rbf', probability=True)

params_svm = {
    "svm__C" : [0.1, 1, 10, 100, 1000],
    "svm__gamma" : [2e-5, 2e-3, 2e-1, "auto", "scale"]
}

pipe_svm = Pipeline([
    ("scaler", StandardScaler()), # normalização
    ("svm", svm)
])

print("fazendo validação svm")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelo_svm = GridSearchCV(pipe_svm, params_svm, cv=cv_strategy, n_jobs=-1)
modelo_svm.fit(X_treino, y_treino)

melhor_param_svm = modelo_svm.best_params_
print("Melhores parâmetros: ", melhor_param_svm)

acuracia_svm = accuracy_score(y_teste, modelo_svm.predict(X_teste)) 
print(f"Acurácia: {acuracia_svm}")

print(classification_report(y_teste, modelo_svm.predict(X_teste)))

ConfusionMatrixDisplay.from_estimator(modelo_svm, X_teste, y_teste)
plt.show()