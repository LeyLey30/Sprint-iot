import cv2
import os
import numpy as np
import json

# ============================
# PARÂMETROS AJUSTÁVEIS
# ============================
DATA_DIR = "data"
MODEL_PATH = "model.yml"
NAMES_FILE = "names.json"

CAMERA_INDEX = 0

# Detecção com Haar Cascade
FACE_SCALE_FACTOR = 1.3
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (100, 100)

# LBPH (reconhecimento)
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8

CONFIDENCE_THRESHOLD = 80.0
# ============================

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ---------------------------------------------
# UTIL: garantir que a pasta exista
# ---------------------------------------------
def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


# ---------------------------------------------
# UTIL: carregar ou criar arquivo de nomes
# ---------------------------------------------
def load_names():
    if not os.path.exists(NAMES_FILE):
        return {}

    with open(NAMES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_name(user_id, name):
    names = load_names()
    names[str(user_id)] = name

    with open(NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=4, ensure_ascii=False)


# ---------------------------------------------
# CAPTURA DE FACES
# ---------------------------------------------
def capture_faces():
    ensure_data_dir()

    try:
        user_id = int(input("Digite o ID numérico para este usuário (ex: 1): "))
    except ValueError:
        print("ID inválido.")
        return

    user_name = input("Digite o nome do usuário (ex: Gustavo): ").strip()
    if not user_name:
        print("Nome inválido.")
        return

    # Salvar nome no sistema
    save_name(user_id, user_name)
    print(f"Nome salvo: ID {user_id} → {user_name}")

    try:
        num_samples = int(input("Quantas imagens deseja capturar? (ex: 50): "))
    except ValueError:
        print("Número inválido.")
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return

    print("Iniciando captura. Olhe para a câmera...")
    print("Pressione 'q' para sair cedo.")

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            filename = f"user.{user_id}.{count}.jpg"

            cv2.imwrite(os.path.join(DATA_DIR, filename), face_img)

            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)
            cv2.putText(frame, f"{count}/{num_samples}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if count >= num_samples:
                print("Coleta concluída.")
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow("Captura de Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------
# TREINAMENTO DO MODELO
# ---------------------------------------------
def load_images_and_labels():
    faces = []
    labels = []

    for file in os.listdir(DATA_DIR):
        if file.lower().endswith(".jpg"):
            parts = file.split(".")
            if len(parts) < 3:
                continue
            label = int(parts[1])
            img = cv2.imread(os.path.join(DATA_DIR, file), cv2.IMREAD_GRAYSCALE)

            if img is not None:
                faces.append(img)
                labels.append(label)

    return faces, labels


def train_model():
    faces, labels = load_images_and_labels()

    if len(faces) == 0:
        print("Nenhuma imagem encontrada.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y
    )

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    print(f"Modelo treinado e salvo em {MODEL_PATH}.")


# ---------------------------------------------
# RECONHECIMENTO EM TEMPO REAL
# ---------------------------------------------
def recognize():
    if not os.path.exists(MODEL_PATH):
        print("Modelo não encontrado.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    names = load_names()

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    print("Reconhecimento iniciado. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label_id, confidence = recognizer.predict(face_img)

            if confidence < CONFIDENCE_THRESHOLD:
                name = names.get(str(label_id), f"ID {label_id}")
                color = (0,255,0)
            else:
                name = "Desconhecido"
                color = (0,0,255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.1f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Reconhecimento Facial", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------
# MENU PRINCIPAL
# ---------------------------------------------
def main():
    while True:
        print("\n=== MENU RECONHECIMENTO FACIAL ===")
        print("1 - Capturar faces")
        print("2 - Treinar modelo")
        print("3 - Reconhecer em tempo real")
        print("0 - Sair")
        choice = input("Escolha uma opção: ")

        if choice == "1":
            capture_faces()
        elif choice == "2":
            train_model()
        elif choice == "3":
            recognize()
        elif choice == "0":
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()
