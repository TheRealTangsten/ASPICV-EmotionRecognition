import os
import re
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
#base_path = curr_path + "\\Date" #OLD
base_path = curr_path + "\\CK\\Cohn_Kanade_images"
base_path_annotation = curr_path + "\\CK\\Cohn Kanade_annotations"

number_emotions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Classes mapping
EMOTION_MAP = {
    'happy': 'positive', #
    'surprise': 'positive', #
    'sadness': 'negative', #
    'anger': 'negative', #
    'fear': 'negative', #
    'disgust': 'negative', #
    'contempt': 'negative', #
}
LETTER_MAP = {
    'F' : 'fear', # 1st - fear, 2nd - ignore, what emotion is that?????
    'S' : 'surprise', # 1st - surprise, 2nd - sad
    'H' : 'happy',
    'G' : 'contempt',
    'D' : 'disgust',
    'A' : 'anger'
}
letter_map_count = {'F':0, 'S':0, 'H':0, 'D':0, 'A': 0, 'G':0}
number_to_letter_map = {'001' : None, '002' : None, '003' : None, '004' : None, '005' : None, '006' : None, '007' : None, '008' : None, '009' : None}

def final_labeling(label, letter_dict, key, letter_count_dict):
    if letter_dict[key] == 'F':
        if letter_count_dict['F'] > 1:
            label = None
    if letter_dict[key] == 'S':
        if letter_count_dict['S'] > 1:
            label = 'sadness'
    return label

# Same version, but with prints enabled
def debug_load_ck_plus_data(base_path):
    data = []
    labels = []
    total_images = 0
    for person_folder in os.listdir(base_path):
        person_path = os.path.join(base_path, person_folder)  # path to all emotions of person
        person_annotation_path = os.path.join(base_path_annotation,
                                              person_folder)  # path to all annotations of emotions of person
        print("\n\n" + person_path + " | ")

        number_to_letter_map = {'001': None, '002': None, '003': None, '004': None, '005': None, '006': None,
                                '007': None, '008': None, '009': None}
        for annotation_tag in os.listdir(person_annotation_path):  # build mapping number-emotion
            number = re.search(r'\d+', annotation_tag)[0]
            letter = re.search(r'[A-Z]', annotation_tag)[0]
            print(number, letter)
            number_to_letter_map[number] = letter
        print(number_to_letter_map)
        letter_map_count = {'F': 0, 'S': 0, 'H': 0, 'D': 0, 'G': 0, 'A': 0}

        # count_emotes = 0
        for numbered_emotion in os.listdir(person_path):
            print("\t " + numbered_emotion)
            #    count_emotes += 1
            # print(f"nr_emotes:{count_emotes}")
            # number_emotions[count_emotes] += 1
            # print("\n\n",number_emotions)
            letter_map_count[number_to_letter_map[numbered_emotion]] += 1
            image_path = os.path.join(person_path, numbered_emotion)
            label1 = number_to_letter_map[numbered_emotion]
            label2 = LETTER_MAP.get(label1, None)
            label3 = final_labeling(label2, number_to_letter_map, numbered_emotion, letter_map_count)
            label = EMOTION_MAP.get(label3, None)
            if label:
                print(f"Image path: {image_path}")
                # print(label + " <- " + label2 + " <- " + label1)
                print(f"\t{label} <- {label3} <- {label2} <- {label1}")
                count_images = 0
                image_path_vector = []
                for img in os.listdir(image_path):
                    image_path_vector.append(os.path.join(image_path, img))
                    count_images += 1
                image_path_vector_index = int(count_images * 0.5) # discard low-intensity complexions
                total_images += count_images - image_path_vector_index
                print(f"Number of images in this category: {count_images} | Start index: {image_path_vector_index}")
                for i in range(image_path_vector_index, len(image_path_vector)):
                    image = cv2.imread(image_path_vector[i], cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        lbp = local_binary_pattern(image, N_POINTS, RADIUS, METHOD)
                        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), density=True)
                        data.append(hist)
                        labels.append(1 if label == 'positive' else 0)
    print(f"Total Images taken into account: {total_images}")
    return np.array(data), np.array(labels)
# Load and preprocess data
def load_ck_plus_data(base_path):
    data = []
    labels = []
    total_images = 0
    for person_folder in os.listdir(base_path):
        person_path = os.path.join(base_path, person_folder) # path to all emotions of person
        person_annotation_path = os.path.join(base_path_annotation, person_folder) # path to all annotations of emotions of person
        print("Parsing \"" + person_path + "\" ...")

        number_to_letter_map = {'001': None, '002': None, '003': None, '004': None, '005': None, '006': None,
                                '007': None, '008': None, '009': None}
        for annotation_tag in os.listdir(person_annotation_path): # build mapping number-emotion
            number = re.search(r'\d+', annotation_tag)[0]
            letter = re.search(r'[A-Z]', annotation_tag)[0]
            #print(number,letter)
            number_to_letter_map[number] = letter
        #print(number_to_letter_map)
        letter_map_count = {'F': 0, 'S': 0, 'H': 0, 'D': 0, 'G':0, 'A':0}

        #count_emotes = 0
        for numbered_emotion in os.listdir(person_path):
            #print("\t " + numbered_emotion)
        #    count_emotes += 1
        #print(f"nr_emotes:{count_emotes}")
        #number_emotions[count_emotes] += 1
        #print("\n\n",number_emotions)
            letter_map_count[number_to_letter_map[numbered_emotion]] += 1
            image_path = os.path.join(person_path, numbered_emotion)
            label1 = number_to_letter_map[numbered_emotion]
            label2 = LETTER_MAP.get(label1, None)
            label3 = final_labeling(label2, number_to_letter_map, numbered_emotion, letter_map_count)
            label = EMOTION_MAP.get(label3, None)
            if label:
                #print(f"Image path: {image_path}")
                #print(label + " <- " + label2 + " <- " + label1)
                #print(f"\t{label} <- {label3} <- {label2} <- {label1}")
                count_images = 0
                image_path_vector = []
                for img in os.listdir(image_path):
                    image_path_vector.append(os.path.join(image_path,img))
                    count_images += 1
                image_path_vector_index = int(count_images * 0.5)
                total_images += count_images - image_path_vector_index
                #print(f"Number of images in this category: {count_images} | Start index: {image_path_vector_index}")
                for i in range(image_path_vector_index, len(image_path_vector)):
                    image = cv2.imread(image_path_vector[i], cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        lbp = local_binary_pattern(image, N_POINTS, RADIUS, METHOD)
                        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), density=True)
                        data.append(hist)
                        labels.append(1 if label == 'positive' else 0)
    #print(f"Total Images taken into account: {total_images}")
    return np.array(data), np.array(labels)


def main():
    # Load data
    data, labels = load_ck_plus_data(base_path)
    print(f"Data size: {np.size(data)} | Labels size: {np.size(labels)}")
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
    #load_ck_plus_data(base_path)
    #print(curr_path)
