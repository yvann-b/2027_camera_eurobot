import time
from bus_zmq.zmq_bus import Publisher
from p_aruco.tag_extractor import TagExtractor
from p_aruco.calibration import TableCalibration
from config import CameraConfig
from p_math.redressement_image import calculer_homographie_via_pose
from ring.frame_ring import FrameRing
from config.loader import CalibrationConfig, CalibrationData, ZoneHeader, Zone, Homographie_config
from p_aruco.tag_extractor import TagExtractor
from p_math.changement_repere import from_coordinate_to_pixel
from zones.save_build import save_zone_visibility
import multiprocessing
import cv2
import traceback
import os
import time

def process_initialisation(frame_ring : FrameRing, param_calib : CalibrationConfig, camera_config : CameraConfig,calib_frame : CalibrationData, start_event : multiprocessing.Event, zones_header : dict[int, ZoneHeader], pub_i_to_app : Publisher , homographie_config : Homographie_config):
    try:
        tag_extractor = TagExtractor()
        tab_calib = TableCalibration(tag_extractor, calib_frame.camera_matrix, calib_frame.dist_coeffs)
        last_frame_id = 1
        liste_zones = []
        for key in zones_header :
            liste_zones.append(zones_header[key])



        # ================================================
        # Attente d'une entrée clavier pour rentrer dans la phase de calibration
        # ================================================

        while True :
            F_mes = frame_ring.read_last_image()
            if F_mes is None:
                continue  # rien à afficher pour l'instant, on reboucle

            frame_visu = cv2.resize(F_mes.frame, (720, int(720 * camera_config.height/camera_config.width)), interpolation=cv2.INTER_LINEAR)

            # Ajout d'un texte indicatif pour le technicien
            cv2.putText(frame_visu, "Appuyez sur une touche pour calibrer...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            windows_waiting = cv2.imshow("frame calib", frame_visu)

            # cv2.waitKey(1) attend 1ms. S'il n'y a pas de touche pressée, il renvoie -1 (soit 255 avec le masque 0xFF)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                print("Touche pressée ! Début de la prise des frames de calibration.")
                cv2.destroyWindow("frame calib")
                break  # On sort de la boucle d'attente

        # ==========================================


        # ================================================
        # Calibration : lecture des 4 tags de référence sur la table
        # ================================================
        attempts = 0

        while tab_calib.valid_frame_count < param_calib.number_frame and attempts < param_calib.number_try_to_calib:
            F_mes = frame_ring.read_last_image()
            last_frame_id = F_mes.frame_id
            frame = F_mes.frame
            tab_calib.analyse_frame(frame)
            attempts += 1
            # Affichage du résultat
            frame_visu = cv2.resize(F_mes.frame, (720, int(720 * camera_config.height/camera_config.width)), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Calibrating ...", frame_visu)

            key = cv2.waitKey(1)

        if tab_calib.valid_frame_count < param_calib.number_frame:
            raise RuntimeError(
                f"Calibration échouée : {tab_calib.valid_frame_count}/{param_calib.number_frame} "
                f"frames valides après {attempts} tentatives"
            )

        pose = tab_calib.finalize()
        print(pose)
        os.makedirs(param_calib.output_calib_file, exist_ok=True)               # Pour créer le dossier s'il n'existe pas déjà
        chemin_complet1 = os.path.join(param_calib.output_calib_file, "pose_cam.npz")
        chemin_complet2 = os.path.join(param_calib.output_calib_file, "zones_visiblity.npz")

        pose.save(chemin_complet1)
        save_zone_visibility(liste_zones, R_cam= pose.R_cam,  T_cam= pose.t_cam, path=chemin_complet2, roi = calib_frame.roi, cam_mtx=calib_frame.camera_matrix )
        H, width, height = calculer_homographie_via_pose(
            new_camera_mtx= calib_frame.camera_matrix,
            R_cam = pose.R_cam,
            t_cam = pose.t_cam,
            homo_config= homographie_config
        )

        for _ in range(10):
            pub_i_to_app.send(
                frame_id=last_frame_id,
                timestamp_ns=time.time_ns(),
                data={"Matrice": H.tolist(), "Width": width, "Height": height}      # Tolist pour sérialiser la donnée sinon ça ne passe pas sur le bus ZMQ
            )
            time.sleep(0.1)  # ~1 secondes au total, largement suffisant

        print("[Initialisation] Done !")
        start_event.set()




    except Exception:
        print("CRASH dans process_frame_debug_display :")
        traceback.print_exc()
