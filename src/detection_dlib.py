# Importation des bibliothèques nécessaires
import cv2          # Bibliothèque de vision par ordinateur
import dlib         # Détection faciale et prédiction des points clés
import time         # Gestion du temps et des délais
import imutils      # Utilitaires pour le traitement d'images
import numpy as np  # Calculs numériques et manipulations de tableaux
import pyttsx3      # Synthèse vocale pour les alertes
import tkinter as tk# Création d'interface graphique
from tkinter import ttk  # Composants GUI modernes
from PIL import Image, ImageTk  # Manipulation d'images pour Tkinter
import winsound     # Génération de sons d'alerte
import threading    # Gestion des threads pour exécution parallèle

" INITIALISATION DES MODULES ET PARAMÈTRES "

# Initialisation du détecteur de visage Dlib (algorithme HOG)
detector = dlib.get_frontal_face_detector()
# Charge le modèle de prédiction des 68 points du visage
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Configuration du moteur de synthèse vocale
moteur_vocal = pyttsx3.init()
# Sélection d'une voix française si disponible
voix_francaises = [v for v in moteur_vocal.getProperty('voices') if 'french' in v.languages or 'fr' in v.id.lower()]
if voix_francaises:
    moteur_vocal.setProperty('voice', voix_francaises[0].id)
# Réglage de la vitesse de parole (150 mots/minute)
moteur_vocal.setProperty('rate', 150)

" DÉFINITION DES SEUILS DE DÉTECTION "

SEUIL_EAR = 0.25             # Seuil de fermeture des yeux (Eye Aspect Ratio)
SEUIL_MAR = 0.70             # Seuil d'ouverture de la bouche (Mouth Aspect Ratio)
SEUIL_CLIGNEMENTS = 0.20     # Seuil de détection de début de clignement
SEUIL_BAILLEMENT = 24        # Nombre de frames pour détection de bâillement
SEUIL_DISTANCE_MAIN_VISAGE = 100 # Distance en pixels main-visage pour alerte
SEUIL_CLIGNEMENTS_PAR_MINUTE = 24 # Seuil de clignements/minute pour alerte

" VARIABLES DE SUIVI ET D'ÉTAT "

compteur_yeux_fermes = 0     # Durée cumulative de fermeture des yeux
compteur_baillement = 0      # Durée cumulative d'ouverture de bouche
compteur_clignements = 0     # Nombre de clignements détectés
temps_debut = time.time()    # Horodatage de démarrage du système
vitesse_actuelle = 0         # Vitesse simulée du véhicule
oeil_ouvert = True           # État actuel des paupières
alerte_somnolence_active = False # Drapeau d'alerte de fatigue
temps_fermeture_yeux = 0     # Horodatage de début de fermeture des yeux
clignotement_en_cours = False # État du clignotement d'alerte
temps_premier_clignement = None # Horodatage du premier clignement

" FONCTION DE SYNTHÈSE VOCALE NON BLOQUANTE "

def parler(message):
    # Fait parler le système dans un thread séparé
    def parler_thread():
        # Fonction interne pour le thread de parole
        moteur_vocal.say(message)
        moteur_vocal.runAndWait()
    # Crée et démarre un nouveau thread pour ne pas bloquer l'interface
    thread = threading.Thread(target=parler_thread)
    thread.start()

def clignoter():
    # Fait clignoter la LED d'alerte avec son
    if clignotement_en_cours:
        return
    clignotement_en_cours = True
    # Boucle de clignotement tant que l'alerte est active
    while alerte_somnolence_active:
        canvas_somnolence.itemconfig(led_somnolence, fill="red")  # Allume en rouge
        winsound.Beep(1000, 500)  # Émet un bip
        time.sleep(0.1)
        canvas_somnolence.itemconfig(led_somnolence, fill="grey") # Éteint
        time.sleep(0.1)
    clignotement_en_cours = False

" FONCTIONS DE CALCUL (EAR, MAR, DISTANCE) "

def calculer_ear(eye_points):
    # Calcule le Eye Aspect Ratio (rapport d'ouverture des yeux)
    # Calcul des distances verticales et horizontales
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    ear = (A + B) / (2.0 * C)  # Formule standard EAR
    return ear

def calculer_mar(mouth_points):
    #Calcule le Mouth Aspect Ratio (rapport d'ouverture de bouche)
    A = np.linalg.norm(np.array(mouth_points[1]) - np.array(mouth_points[5]))
    B = np.linalg.norm(np.array(mouth_points[2]) - np.array(mouth_points[4]))
    C = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[3]))
    mar = (A + B) / (2.0 * C)  # Formule standard MAR
    return mar

def calculer_distance(point1, point2):
    #Calcule la distance euclidienne entre deux points
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

"FONCTION DE TRAITEMENT DE L'IMAGE"

def traiter_image(image):
    #Traite l'image pour détecter les signes de fatigue
    global compteur_yeux_fermes, compteur_baillement, compteur_clignements
    global oeil_ouvert, alerte_somnolence_active, temps_fermeture_yeux
    global clignotement_en_cours, temps_premier_clignement

    # Redimensionnement de l'image pour traitement plus rapide
    image = imutils.resize(image, width=400)
    
    # Conversion en niveaux de gris si nécessaire
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Détection des visages dans l'image
    rects = detector(gray, 0)
    
    for rect in rects:
        # Prédiction des points clés du visage
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype="int")
        
        # Extraction des coordonnées des 68 points
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
            cv2.circle(image, (coords[i][0], coords[i][1]), 1, (0, 0, 255), -1)

        # Calcul EAR pour les yeux gauche et droit
        leftEye = coords[36:42]
        rightEye = coords[42:48]
        ear_left = calculer_ear(leftEye)
        ear_right = calculer_ear(rightEye)
        ear = (ear_left + ear_right) / 2.0

        # Calcul MAR pour la bouche
        mouth = [coords[48], coords[50], coords[52], coords[54], coords[56], coords[58]]
        mar = calculer_mar(mouth)

        # Affichage des valeurs sur l'image
        cv2.putText(image, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"MAR: {mar:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Détection de somnolence (yeux fermés)
        if ear < SEUIL_EAR:
            if temps_fermeture_yeux == 0:
                temps_fermeture_yeux = time.time()
            elif time.time() - temps_fermeture_yeux > 3:
                if not alerte_somnolence_active:
                    alerte_somnolence_active = True
                    parler("Attention! Vous semblez fatigué. Concentrez-vous sur la route.")
                    canvas_somnolence.itemconfig(led_somnolence, fill="yellow")
                elif time.time() - temps_fermeture_yeux > 6:
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
                parler("Attention! Vous baillez fréquemment. Prenez une pause.")
                canvas_baillement.itemconfig(led_baillement, fill="red")
        else:
            compteur_baillement = 0
            canvas_baillement.itemconfig(led_baillement, fill="grey")

        # Détection des clignements excessifs
        if ear < SEUIL_CLIGNEMENTS and oeil_ouvert:
            oeil_ouvert = False
        elif ear >= SEUIL_EAR and not oeil_ouvert:
            oeil_ouvert = True
            compteur_clignements += 1
            if temps_premier_clignement is None:
                temps_premier_clignement = time.time()
            duree_ecoulee = time.time() - temps_premier_clignement
            if duree_ecoulee > 60:
                compteur_clignements = 1
                temps_premier_clignement = time.time()
            if compteur_clignements > SEUIL_CLIGNEMENTS_PAR_MINUTE and duree_ecoulee <= 60:
                parler("Attention! Vous clignez des yeux fréquemment. Soyez vigilant.")
                canvas_clignements.itemconfig(led_clignements, fill="red")
                compteur_clignements = 0
        else:
            canvas_clignements.itemconfig(led_clignements, fill="grey")

        cv2.putText(image, f"Clignements: {compteur_clignements}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Marquage du nez pour référence
        nez = coords[30]
        cv2.circle(image, (nez[0], nez[1]), 3, (255, 0, 0), -1)

        break  # Ne traite que le premier visage détecté

    return image

"INTERFACE GRAPHIQUE AVEC TKINTER"

# Création de la fenêtre principale
root = tk.Tk()
root.title("Système ADAS avec Dlib")
root.geometry("800x600")

# Configuration de l'affichage vidéo
label_video = tk.Label(root)
label_video.pack(pady=10)

# Cadre pour les indicateurs LED
cadre_leds = ttk.Frame(root)
cadre_leds.pack(pady=10)

# Création des LED d'état
def creer_led(parent, texte):
    # Crée un indicateur LED avec étiquette
    canvas = tk.Canvas(parent, width=40, height=40)
    canvas.pack(side=tk.LEFT, padx=5)
    led = canvas.create_oval(10, 10, 30, 30, fill="grey")
    ttk.Label(parent, text=texte).pack(side=tk.LEFT, padx=5)
    return canvas, led

# LED pour l'état d'activation
canvas_activation, led_activation = creer_led(cadre_leds, "Activation")
# LED pour l'alerte de somnolence
canvas_somnolence, led_somnolence = creer_led(cadre_leds, "Somnolence")
# LED pour l'alerte de bâillement
canvas_baillement, led_baillement = creer_led(cadre_leds, "Bâillement")
# LED pour l'alerte de clignements
canvas_clignements, led_clignements = creer_led(cadre_leds, "Clignements")

# Cadre pour le contrôle de vitesse
cadre_vitesse = ttk.Frame(root)
cadre_vitesse.pack(pady=10)

# Affichage de la vitesse
label_vitesse = ttk.Label(cadre_vitesse, text="Vitesse: 0 km/h", font=("Helvetica", 14))
label_vitesse.pack(side=tk.LEFT, padx=10)

# Curseur de contrôle de vitesse
curseur_vitesse = ttk.Scale(cadre_vitesse, from_=0, to=120, orient=tk.HORIZONTAL, length=300)
curseur_vitesse.pack(side=tk.LEFT, padx=10)
curseur_vitesse.bind("<ButtonRelease-1>", lambda e: mettre_a_jour_vitesse())

# Bouton de sortie
bouton_quitter = ttk.Button(root, text="Quitter", command=root.quit)
bouton_quitter.pack(pady=10)

# Initialisation du flux vidéo
flux_video = cv2.VideoCapture(0)
if not flux_video.isOpened():
    print("Erreur: Impossible d'accéder à la caméra")
    exit()
time.sleep(2.0)  # Pause pour l'initialisation de la caméra

"BOUCLE PRINCIPALE DE MISE À JOUR"

def mettre_a_jour_image():
    # Met à jour l'image affichée et l'état du système
    global vitesse_actuelle
    ret, image = flux_video.read()
    
    if not ret or image is None:
        print("Erreur : Impossible de capturer l'image.")
        label_video.after(10, mettre_a_jour_image)
        return
    
    if vitesse_actuelle > 20:  # Activation du système
        canvas_activation.itemconfig(led_activation, fill="green")
        image = traiter_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:  # Système désactivé
        canvas_activation.itemconfig(led_activation, fill="grey")
        image = np.zeros((300, 400, 3), dtype=np.uint8)  # Image noire
    
    # Mise à jour de l'affichage
    img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=img)
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)
    label_video.after(10, mettre_a_jour_image)  # Rappel toutes les 10ms

def mettre_a_jour_vitesse():
    # Met à jour l'affichage de la vitesse
    global vitesse_actuelle
    vitesse_actuelle = curseur_vitesse.get()
    label_vitesse.config(text=f"Vitesse: {vitesse_actuelle} km/h")

# Démarrage de la boucle principale
mettre_a_jour_image()
root.mainloop()

"NETTOYAGE FINAL"

flux_video.release()  # Libération de la caméra
cv2.destroyAllWindows()  # Fermeture des fenêtres OpenCV