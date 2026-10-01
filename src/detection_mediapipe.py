# Importation des bibliothèques nécessaires
import cv2  # Pour le traitement d'image
import time  # Pour gérer le temps
import numpy as np  # Pour les calculs mathématiques
import mediapipe as mp  # Pour la détection des visages et mains
from imutils.video import VideoStream  # Pour capturer la vidéo
import pyttsx3  # Pour la synthèse vocale
import tkinter as tk  # Pour l'interface graphique
from tkinter import ttk  # Pour les widgets modernes
from PIL import Image, ImageTk  # Pour afficher les images
import winsound  # Pour les alertes sonores
import threading  # Pour exécuter des tâches en parallèle
import cvzone  # Pour des fonctions utiles de vision par ordinateur
from cvzone.FaceDetectionModule import FaceDetector  # Pour détecter les visages

"INITIALISATION DES MODULES"

# Initialisation des modèles MediaPipe pour le visage et les mains
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Configuration de la détection du visage
maillage_visage = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Détecte maximum 1 visage
    refine_landmarks=True,  # Améliore la précision
    min_detection_confidence=0.5,  # Seuil minimum de confiance
    min_tracking_confidence=0.5  # Seuil minimum pour suivre le visage
)

# Configuration de la détection des mains
mains = mp_hands.Hands(
    max_num_hands=2,  # Détecte maximum 2 mains
    min_detection_confidence=0.7  # Seuil minimum de confiance
)

# Initialisation du moteur de synthèse vocale
moteur_vocal = pyttsx3.init()

# Sélection d'une voix française 
voix_francaises = [v for v in moteur_vocal.getProperty('voices') if 'french' in v.languages or 'fr' in v.id.lower()]
if voix_francaises:
    moteur_vocal.setProperty('voice', voix_francaises[0].id)
moteur_vocal.setProperty('rate', 170)  # Vitesse de parole

# Initialisation du détecteur de visage
detecteur_visage = FaceDetector(minDetectionCon=0.7)  # Seuil de confiance à 70%

"PARAMÈTRES DU SYSTÈME"

# Seuils pour les différentes détections
SEUIL_EAR = 0.25  # Seuil pour les yeux fermés
SEUIL_MAR = 0.30  # Seuil pour la bouche ouverte (bâillement)
SEUIL_CLIGNEMENTS = 0.23  # Seuil pour détecter un clignement
SEUIL_BAILLEMENT = 24  # Nombre de frames pour détecter un bâillement
SEUIL_DISTANCE_MAIN_VISAGE = 150  # Distance en pixels pour alerter si main trop proche
SEUIL_CLIGNEMENTS_PAR_MINUTE = 24  # Nombre max de clignements par minute

# Paramètres pour le calcul du rythme cardiaque (BPM)
FS = 30  # Fréquence d'échantillonnage (30 images par seconde)
BUFFER_SIZE = 150  # Tampon pour 5 secondes de données
MIN_HZ = 0.83  # Fréquence minimale (50 battements par minute)
MAX_HZ = 2.5   # Fréquence maximale (150 battements par minute)
MIN_FRAMES = 30  # Nombre minimum d'images pour calculer le BPM
seuil_bpm_bas = 50  # Seuil bas pour alerte BPM
seuil_bpm_haut = 120  # Seuil haut pour alerte BPM

# Variables pour stocker les données du BPM
bpm_buffer = []  # Stocke les mesures d'intensité
timestamps = []  # Stocke les moments des mesures
last_valid_bpm = 0  # Dernière valeur valide du BPM
last_bpm_time = 0  # Moment du dernier BPM valide
current_bpm_display = "---"  # Valeur affichée à l'écran
bpm_stable = None  # Dernière valeur stable du BPM

"VARIABLES DE SUIVI"

compteur_yeux_fermes = 0  # Compte les frames avec yeux fermés
compteur_baillement = 0  # Compte les frames avec bouche ouverte
compteur_clignements = 0  # Compte les clignements
temps_debut = time.time()  # Moment de démarrage du système
vitesse_actuelle = 0  # Vitesse du véhicule
oeil_ouvert = True  # État des yeux (ouvert/fermé)
alerte_somnolence_active = False  # Indique si alerte de somnolence est active
temps_fermeture_yeux = 0  # Moment où les yeux se ferment
clignotement_en_cours = False  # Indique si l'alerte clignote
temps_premier_clignement = None  # Moment du premier clignement
fps = 0  # Images par seconde
compteur_frames = 0  # Compte les frames traitées
temps_debut_fps = time.time()  # Début du calcul des FPS
dernier_alerte_bpm = 0  # Moment de la dernière alerte BPM
systeme_active_precedent = None  # État précédent du système (activé/désactivé)
main_proche_active = False  # Indique si une main est proche du visage

"FONCTIONS PRINCIPALES "
def parler(message):
    "Fait parler le système avec le message donné"
    def parler_thread():
        moteur_vocal.say(message)
        moteur_vocal.runAndWait()
    threading.Thread(target=parler_thread, daemon=True).start()

def clignoter():
    "Fait clignoter la LED et émettre un son d'alerte"
    global clignotement_en_cours
    if clignotement_en_cours:
        return
    clignotement_en_cours = True
    try:
        while alerte_somnolence_active:
            canvas_somnolence.itemconfig(led_somnolence, fill="red")
            winsound.Beep(1000, 500)  # Bip sonore
            time.sleep(0.1)
            canvas_somnolence.itemconfig(led_somnolence, fill="grey")
            time.sleep(0.1)
    finally:
        clignotement_en_cours = False

def calculer_ear(landmarks_oeil):
    "Calcule le rapport d'ouverture des yeux (EAR)"
    A = np.linalg.norm(np.array(landmarks_oeil[1]) - np.array(landmarks_oeil[5]))
    B = np.linalg.norm(np.array(landmarks_oeil[2]) - np.array(landmarks_oeil[4]))
    C = np.linalg.norm(np.array(landmarks_oeil[0]) - np.array(landmarks_oeil[3]))
    return (A + B) / (2.0 * C)

def calculer_mar(landmarks_bouche):
    "Calcule le rapport d'ouverture de la bouche (MAR)"
    A = np.linalg.norm(np.array(landmarks_bouche[1]) - np.array(landmarks_bouche[5]))
    B = np.linalg.norm(np.array(landmarks_bouche[2]) - np.array(landmarks_bouche[4]))
    C = np.linalg.norm(np.array(landmarks_bouche[0]) - np.array(landmarks_bouche[3]))
    return (A + B) / (2.0 * C)

def calculer_distance(point1, point2):
    "Calcule la distance entre deux points"
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_bpm(face_roi):
    "Calcule le rythme cardiaque (BPM) à partir d'une région du visage"
    global bpm_buffer, timestamps, last_valid_bpm, last_bpm_time, bpm_stable
    
    # Convertir l'image en niveaux de gris et normaliser
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    avg_intensity = np.mean(gray)
    
    # Ajouter les données aux tampons
    timestamp = time.time()
    bpm_buffer.append(avg_intensity)
    timestamps.append(timestamp)
    
    # Garder seulement les dernières mesures
    if len(bpm_buffer) > BUFFER_SIZE:
        bpm_buffer = bpm_buffer[-BUFFER_SIZE:]
        timestamps = timestamps[-BUFFER_SIZE:]
    
    # Calculer seulement si assez de données
    if len(bpm_buffer) >= MIN_FRAMES:
        try:
            # Traitement du signal
            signal = np.array(bpm_buffer)
            signal = (signal - np.mean(signal)) / np.std(signal)
            
            # Analyse fréquentielle (FFT)
            n = len(signal)
            fps = float(n) / (timestamps[-1] - timestamps[0])
            fft = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(n, d=1.0/fps)
            
            # Filtrage des fréquences
            mask = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
            fft[~mask] = 0
            
            # Trouver le pic principal
            peak_idx = np.argmax(fft)
            bpm = freqs[peak_idx] * 60.0  # Convertir en battements par minute
            
            # Vérifier si le BPM est physiologiquement plausible
            if 50 <= bpm <= 150:
                last_valid_bpm = bpm
                last_bpm_time = time.time()
                bpm_stable = bpm
                return bpm
                
        except Exception as e:
            print(f"Erreur calcul BPM: {e}")
    
    # Retourner la dernière valeur valide si récente 
    if time.time() - last_bpm_time < 8:
        return bpm_stable
    return None

def traiter_rythme_cardiaque(image):
    "Détecte le visage et calcule le rythme cardiaque"
    global current_bpm_display
    
    # Détecter les visages dans l'image
    image_visage, visages = detecteur_visage.findFaces(image.copy(), draw=False)
    
    if not visages:
        # Si pas de visage mais dernière valeur valide récente
        if time.time() - last_bpm_time < 8 and bpm_stable is not None:
            current_bpm_display = str(int(bpm_stable))
            return bpm_stable
        current_bpm_display = "---"
        return None
    
    # Prendre le visage principal (le plus grand)
    visage = max(visages, key=lambda v: v['bbox'][2]*v['bbox'][3])
    x, y, w, h = [int(v) for v in visage['bbox']]
    
    # Définir la zone d'intérêt avec une marge
    margin = int(w * 0.3)
    x, y = max(0, x-margin), max(0, y-margin)
    w, h = min(w+2*margin, image.shape[1]-x), min(h+2*margin, image.shape[0]-y)
    
    face_roi = image[y:y+h, x:x+w]
    
    # Vérifier si la zone est valide
    if face_roi.size == 0 or np.mean(face_roi) < 30:
        if time.time() - last_bpm_time < 8 and bpm_stable is not None:
            current_bpm_display = str(int(bpm_stable))
            return bpm_stable
        current_bpm_display = "---"
        return None
    
    # Calculer le BPM
    bpm = calculate_bpm(face_roi)
    if bpm is not None:
        current_bpm_display = str(int(bpm))
    else:
        current_bpm_display = "---"
    
    return bpm

def traiter_image(image):
    "Traite une image pour détecter les différents signes"
    global compteur_yeux_fermes, compteur_baillement, compteur_clignements
    global oeil_ouvert, alerte_somnolence_active, temps_fermeture_yeux
    global fps, compteur_frames, temps_debut_fps, dernier_alerte_bpm
    global main_proche_active, temps_premier_clignement 

    # Calcul des images par seconde (FPS)
    compteur_frames += 1
    if time.time() - temps_debut_fps >= 1.0:
        fps = compteur_frames
        compteur_frames = 0
        temps_debut_fps = time.time()

    # Redimensionner l'image et la convertir en RGB
    image = cv2.resize(image, (400, 300))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculer le BPM toutes les 2 images
    bpm = None
    if compteur_frames % 2 == 0:
        bpm = traiter_rythme_cardiaque(image.copy())
    
    # Réinitialiser l'état de détection des mains
    main_proche_active = False
    
    # Détection des visages avec MediaPipe
    resultats_visage = maillage_visage.process(image_rgb)
    nez_x, nez_y = None, None

    if resultats_visage.multi_face_landmarks:
        for landmarks_visage in resultats_visage.multi_face_landmarks:
            # Extraire les points clés des yeux et de la bouche
            landmarks_oeil_gauche = [landmarks_visage.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            landmarks_oeil_droit = [landmarks_visage.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            landmarks_bouche = [landmarks_visage.landmark[i] for i in [61, 82, 312, 291, 317, 87]]

            # Convertir les coordonnées normalisées en pixels
            points_oeil_gauche = [(int(p.x * image.shape[1]), int(p.y * image.shape[0])) for p in landmarks_oeil_gauche]
            points_oeil_droit = [(int(p.x * image.shape[1]), int(p.y * image.shape[0])) for p in landmarks_oeil_droit]
            points_bouche = [(int(p.x * image.shape[1]), int(p.y * image.shape[0])) for p in landmarks_bouche]

            # Calculer les rapports d'ouverture
            ear = (calculer_ear(points_oeil_gauche) + calculer_ear(points_oeil_droit)) / 2
            mar = calculer_mar(points_bouche)

            # Afficher les informations sur l'image
            cv2.putText(image, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"FPS: {fps}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Afficher le BPM
            cv2.putText(image, f"BPM: {current_bpm_display}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (255, 255, 255) if current_bpm_display != "---" else (0, 0, 255), 2)
            
            # Gérer les alertes BPM
            # if bpm is not None:
            #     if bpm < seuil_bpm_bas or bpm > seuil_bpm_haut:
            #         canvas_rythme.itemconfig(led_rythme, fill="red")
            #         if time.time() - dernier_alerte_bpm > 7:
            #             parler("Alerte: rythme cardiaque anormal!")
            #             dernier_alerte_bpm = time.time()
            #     else:
            #         canvas_rythme.itemconfig(led_rythme, fill="green")
            # elif current_bpm_display == "---":
            #     canvas_rythme.itemconfig(led_rythme, fill="grey")

            # Détection de somnolence (yeux fermés)
            if ear < SEUIL_EAR:
                if temps_fermeture_yeux == 0:
                    temps_fermeture_yeux = time.time()
                elif time.time() - temps_fermeture_yeux > 3:
                    if not alerte_somnolence_active:
                        alerte_somnolence_active = True
                        parler("Attention! Vous semblez fatigué.")
                        canvas_somnolence.itemconfig(led_somnolence, fill="yellow")
                    elif time.time() - temps_fermeture_yeux > 7:
                        if not clignotement_en_cours:
                            threading.Thread(target=clignoter, daemon=True).start()
            else:
                temps_fermeture_yeux = 0
                if alerte_somnolence_active:
                    alerte_somnolence_active = False
                    canvas_somnolence.itemconfig(led_somnolence, fill="grey")

            # Détection des bâillements
            if mar > SEUIL_MAR:
                compteur_baillement += 1
                if compteur_baillement > SEUIL_BAILLEMENT:
                    parler("Attention! Vous baillez fréquemment.")
                    canvas_baillement.itemconfig(led_baillement, fill="red")
            else:
                compteur_baillement = 0
                canvas_baillement.itemconfig(led_baillement, fill="grey")

            # Détection des clignements
            if ear < SEUIL_CLIGNEMENTS and oeil_ouvert:
                oeil_ouvert = False
                if temps_premier_clignement is None:
                    temps_premier_clignement = time.time()

            elif ear >= SEUIL_EAR and not oeil_ouvert:
                oeil_ouvert = True
                compteur_clignements += 1
    
                # Si trop de clignements
                if compteur_clignements >= SEUIL_CLIGNEMENTS_PAR_MINUTE:
                    parler("Attention! Vous clignez des yeux fréquemment.")
                    canvas_clignements.itemconfig(led_clignements, fill="red")
                    compteur_clignements = 0
                    temps_premier_clignement = None
    
                # Initialiser ou réinitialiser le timer
                if temps_premier_clignement is None:
                    temps_premier_clignement = time.time()
                else:
                    # Réinitialiser après 1 minute
                    duree_ecoulee = time.time() - temps_premier_clignement
                    if duree_ecoulee > 60:
                        compteur_clignements = 1
                        temps_premier_clignement = time.time()
            else:
                canvas_clignements.itemconfig(led_clignements, fill="grey")

            # Afficher le compteur de clignements
            cv2.putText(image, f"Clignements: {compteur_clignements}", (10, 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Coordonnées du nez pour détection des mains
            nez = landmarks_visage.landmark[1]
            nez_x, nez_y = int(nez.x * image.shape[1]), int(nez.y * image.shape[0])

    # Détection des mains
    resultats_mains = mains.process(image_rgb)
    if resultats_mains.multi_hand_landmarks:
        for landmarks_main in resultats_mains.multi_hand_landmarks:
            poignet = landmarks_main.landmark[0]
            poignet_x, poignet_y = int(poignet.x * image.shape[1]), int(poignet.y * image.shape[0])
            cv2.circle(image, (poignet_x, poignet_y), 5, (0, 255, 0), -1)

            # Calculer la distance main-nez
            if nez_x is not None and nez_y is not None:
                distance = calculer_distance((nez_x, nez_y), (poignet_x, poignet_y))
                cv2.putText(image, f"Distance: {distance:.2f}", (10, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Alerte si main trop proche
                if distance < SEUIL_DISTANCE_MAIN_VISAGE:
                    main_proche_active = True
                    cv2.putText(image, "ALERTE : Main proche !", (10, 210), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    parler("Attention! Gardez vos mains sur le volant.")

    # Mise à jour de la LED pour la main proche
    if main_proche_active:
        canvas_main_proche.itemconfig(led_main_proche, fill="red")
    else:
        canvas_main_proche.itemconfig(led_main_proche, fill="grey")

    return image

"INTERFACE GRAPHIQUE"

# Création de la fenêtre principale
root = tk.Tk()
root.title("Système ADAS Amélioré")
root.geometry("900x650")  # Taille de la fenêtre

# Zone d'affichage vidéo
label_video = tk.Label(root)
label_video.pack(pady=10)

# Cadre pour les LEDs d'état
cadre_leds = ttk.Frame(root)
cadre_leds.pack(pady=10)

def creer_led(parent, texte):
    "Crée une LED avec un label descriptif"
    canvas = tk.Canvas(parent, width=40, height=40)
    canvas.pack(side=tk.LEFT, padx=5)
    led = canvas.create_oval(10, 10, 30, 30, fill="grey")
    ttk.Label(parent, text=texte).pack(side=tk.LEFT, padx=5)
    return canvas, led

# Création des différentes LEDs
canvas_activation, led_activation = creer_led(cadre_leds, "Activation")
canvas_somnolence, led_somnolence = creer_led(cadre_leds, "Somnolence")
canvas_baillement, led_baillement = creer_led(cadre_leds, "Bâillement")
canvas_clignements, led_clignements = creer_led(cadre_leds, "Clignements")
canvas_rythme, led_rythme = creer_led(cadre_leds, "Rythme Cardiaque")
canvas_main_proche, led_main_proche = creer_led(cadre_leds, "Main proche")

# Cadre pour le contrôle de vitesse
cadre_vitesse = ttk.Frame(root)
cadre_vitesse.pack(pady=10)

# Affichage de la vitesse
label_vitesse = ttk.Label(cadre_vitesse, text="Vitesse: 0 km/h", font=("Helvetica", 14))
label_vitesse.pack(side=tk.LEFT, padx=10)

# Curseur pour régler la vitesse
curseur_vitesse = ttk.Scale(cadre_vitesse, from_=0, to=120, orient=tk.HORIZONTAL, length=300)
curseur_vitesse.pack(side=tk.LEFT, padx=10)

# Bouton pour quitter l'application
bouton_quitter = ttk.Button(root, text="Quitter", command=root.quit)
bouton_quitter.pack(pady=10)

##### FONCTIONS DE MISE À JOUR #####

def mettre_a_jour_vitesse():
    "Met à jour l'affichage de la vitesse"
    global vitesse_actuelle
    vitesse_actuelle = curseur_vitesse.get()
    label_vitesse.config(text=f"Vitesse: {int(vitesse_actuelle)} km/h")

# Lier le curseur à la fonction de mise à jour
curseur_vitesse.bind("<ButtonRelease-1>", lambda e: mettre_a_jour_vitesse())

def mettre_a_jour_image():
    "Met à jour l'image affichée et gère l'état du système"""
    global systeme_active_precedent
    
    # Vérifier si le système doit être activé (vitesse > 20 km/h)
    if vitesse_actuelle > 20:
        # Activer le système
        canvas_activation.itemconfig(led_activation, fill="green")
        image = flux_video.read()
        
        if image is not None:
            try:
                # Traiter et afficher l'image
                image_traitee = traiter_image(image)
                img = Image.fromarray(cv2.cvtColor(image_traitee, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                label_video.imgtk = imgtk
                label_video.configure(image=imgtk)
            except Exception as e:
                print(f"Erreur traitement image: {e}")
        
        # Annoncer l'activation si changement d'état
        if systeme_active_precedent != True:
            parler("Système ADAS activé")
            systeme_active_precedent = True
    else:
        # Désactiver le système
        canvas_activation.itemconfig(led_activation, fill="grey")
        canvas_rythme.itemconfig(led_rythme, fill="grey")
        canvas_main_proche.itemconfig(led_main_proche, fill="grey")
        
        # Afficher une image noire
        img_noire = np.zeros((300, 400, 3), dtype=np.uint8)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img_noire))
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)
        
        # Annoncer la désactivation si changement d'état
        if systeme_active_precedent != False:
            parler("Système ADAS désactivé")
            systeme_active_precedent = False
    
    # Planifier la prochaine mise à jour
    label_video.after(30, mettre_a_jour_image)

##### LANCEMENT #####

# Démarrer la capture vidéo
flux_video = VideoStream(src=0).start()
time.sleep(2.0)  # Laisser le temps à la caméra de s'initialiser

# Démarrer la mise à jour de l'interface
mettre_a_jour_image()

# Lancer la boucle principale de l'interface
root.mainloop()

# Nettoyage à la fermeture
cv2.destroyAllWindows()
flux_video.stop()