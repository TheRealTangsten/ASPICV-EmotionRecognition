import os
import cv2 as cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import local_binary_pattern

# Parameters for LBP
RADIUS = 1
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

curr_path = os.path.abspath(os.getcwd())
#base_path = "C:\\Users\\Tangsten\\PycharmProjects\\ASPICV_EmotionRecognition\\Date" # Adapteaza path-ul la statia curenta.
base_path = curr_path + "\\Date" # Adapteaza path-ul la statia curenta.

# Classes mapping
EMOTION_MAP = {
    'happy': 'positive',
    'surprise': 'positive',
    'sadness': 'negative',
    'anger': 'negative',
    'fear': 'negative',
    'disgust': 'negative',
    'contempt': 'negative',
}

# Load and preprocess data
def load_ck_plus_data(base_path):
    data = []
    labels = []
    for emotion_folder in os.listdir(base_path):
        emotion_path = os.path.join(base_path, emotion_folder)
        #print("\n\n" + emotion_path + " | ")
        if os.path.isdir(emotion_path):
            label = EMOTION_MAP.get(emotion_folder, None)
            #print(label)
            if label:
                person_folders = sorted(os.listdir(emotion_path))
                #print(person_folders)
                for person_folder in person_folders:
                    person_path = os.path.join(emotion_path, person_folder)
                    #print(person_path)
                    img_path = person_path
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        lbp = local_binary_pattern(img, N_POINTS, RADIUS, METHOD)
                        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), density=True)
                        data.append(hist)
                        labels.append(1 if label == 'positive' else 0)
    return np.array(data), np.array(labels)


def main():
    # Load data
    data, labels = load_ck_plus_data(base_path)

    #print(data)

    # Normalize features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Hiperparametri
    hidden_layer_sizes = [50, 100, 500]#[50, 100, 500, 1000, 5000]
    #se poate observa un trend crescator al performantei in relatie cu numarul de neuroni de pe stratul ascuns
    #performanta poate creste ori cu numarul de neuroni, ori prin adaugarea a mai multor straturi ascunse.
    learning_rates = [0.001, 0.01, 1]

    # Train + test
    best_accuracy = 0
    best_model = None
    for h_size in hidden_layer_sizes:
        for lr in learning_rates:
            print(f"\n------------------------------------------------------------------")
            print(f"MLP cu hidden_layer_size={h_size}, learning_rate={lr}")
            mlp = MLPClassifier(hidden_layer_sizes=(h_size,), learning_rate_init=lr, max_iter=500, random_state=42)
            mlp.fit(X_train, y_train)#train
            y_pred = mlp.predict(X_test)#test
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy*100:.2f}%")
            print(classification_report(y_test, y_pred))
            print(f"------------------------------------------------------------------\n")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_size = h_size
                best_lr = lr
                best_model = mlp

    print(f"Best Model:\nAccuracy: {best_accuracy*100:.2f}%\nLayer Size: {best_size}\nLearnin Rate: {best_lr}")

    # Save the best model (optional)
    if best_model:
        from joblib import dump
        dump(best_model, "best_mlp_model.joblib")

if __name__ == "__main__":
    main()
    #print(curr_path)
