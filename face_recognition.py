import cv2
import numpy as np
import os

# Coloque aqui o ip recebido pelo ESP, ele aparecerá no Monitor
STREAM_URL = "http://192.168.3.37:81/stream"  # Substitua pelo seu IP e porta

# inicializa o detector de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# criando o reconhecedor facial LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# diretório temporário para salvar imagens capturadas
SAVE_DIR = "captured_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

# função para capturar imagens da transmissão ao vivo
def capture_faces(stream_url, person_id, num_samples=20):
    cap = cv2.VideoCapture(stream_url)
    count = 0
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame da câmera.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            filename = f"{SAVE_DIR}/{person_id}_{count}.jpg"
            cv2.imwrite(filename, face)
            count += 1
            print(f"Imagem {count}/{num_samples} salva: {filename}")

            # Desenha um retângulo ao redor do rosto detectado
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("Capturando rostos", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# função para treinar o modelo com as imagens capturadas
def train_model():
    face_samples = []
    face_ids = []
    
    image_paths = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR)]
    
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        person_id = int(os.path.basename(path).split("_")[0])  # Extrai o ID do nome do arquivo
        
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y+h, x:x+w])
            face_ids.append(person_id)
    
    print("Treinando modelo LBPH...")
    recognizer.train(face_samples, np.array(face_ids))
    recognizer.write("trained_model.yml")
    print("Treinamento concluído!")

# função para reconhecimento facial na transmissão ao vivo
def recognize_faces(stream_url):
    recognizer.read("trained_model.yml")
    cap = cv2.VideoCapture(stream_url)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame da câmera.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        for (x, y, w, h) in faces_detected:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_roi)
            print(f"Reconhecido como ID: {label} | Confiança: {confidence:.2f}")

            # Desenha retângulo e exibe ID
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()

# fluxo Principal
def main():
    while True:
        print("\n===== MENU =====")
        print("1 - Capturar novo ID")
        print("2 - Treinar modelo")
        print("3 - Iniciar reconhecimento facial")
        print("4 - Sair")
        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            person_id = input("Digite um ID numérico para a pessoa: ")
            capture_faces(STREAM_URL, person_id, num_samples=20)
        elif opcao == "2":
            train_model()
        elif opcao == "3":
            recognize_faces(STREAM_URL)
        elif opcao == "4":
            print("Saindo do programa...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
