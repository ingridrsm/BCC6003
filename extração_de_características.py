import os
import cv2
import re 
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model                 

# extrair rótulo dos arquivos
def extrair_label(filename):
    match = re.match(r'^[a-zA-Z]+', filename)
    if match:
        return match.group() 
    return 'desconhecido'  

# processar imagens e rótulos
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        label = extrair_label(filename)
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE) 
        if img is not None:
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            images.append(img)
            labels.append(label)
            print(f"Imagem carregada: {filename} | Label: {label}")
    return np.array(images), np.array(labels)

# extrair as características 
def extract_features(images):
    print("EXTRAINDO")
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    images_preprocessadas = preprocess_input(images)
    features = base_model.predict(images_preprocessadas)
    return features

folder = 'meses'

images, labels = load_images_from_folder(folder)

X = extract_features(images)
y = np.array(labels)

# normalização
scaler = StandardScaler()
X = scaler.fit_transform(X)

# salvar nos arquivos
output_dir = 'features'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.save(os.path.join(output_dir, 'X.npy'), X)
np.save(os.path.join(output_dir, 'y.npy'), y)

print("iamgensguardadas")