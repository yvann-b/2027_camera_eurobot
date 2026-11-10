from config.loader import FrameMessage
import multiprocessing
import cv2
import traceback

def process_frame_debug_display(frame_ring,stop_event: multiprocessing.Event):
    try:
        compteur_frame_none = 0
        while (not stop_event.is_set()):
            F_mes = frame_ring.read_last_image()
            if F_mes is None:
                #print(f"Compteur frame : {compteur_frame_none}")
                compteur_frame_none += 1
                continue  # rien à afficher pour l'instant, on reboucle
            frame = F_mes.frame
            # Paramètres du texte
            texte = f"Frame_ID {F_mes.frame_id} | Timestamp {F_mes.frame_timestamp_ns}"
            position = (20, 40)  # (X, Y) : 20px du bord gauche, 40px du haut
            police = cv2.FONT_HERSHEY_SIMPLEX
            echelle = 1.0  # Taille de la police
            couleur = (0, 255, 0)  # Vert au format BGR (Bleu, Vert, Rouge)
            epaisseur = 3

            # Ajout du texte sur la frame
            cv2.putText(frame, texte, position, police, echelle, couleur, epaisseur, cv2.LINE_AA)

            # Affichage du résultat
            frame = cv2.resize(frame, (720, int(720 * 2048/2448)), interpolation=cv2.INTER_LINEAR)

            cv2.imshow(str(frame_ring.name + "resize"), frame)
            key = cv2.waitKey(1)
    except Exception:
        print("CRASH dans process_frame_debug_display :")
        traceback.print_exc()

