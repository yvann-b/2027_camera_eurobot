"""
Registre des canaux du bus ZMQ interne.

CE FICHIER NE CONTIENT QUE DE LA DONNEE.
Aucune socket, aucun zmq.Context n'est créé ici — c'est core/zmq_bus.py qui
s'en charge, et seulement au moment où un module en a réellement besoin.
Ce fichier répond uniquement à la question "quels canaux existent, et
comment doivent-ils être lus", pas "comment on les utilise".
"""
from dataclasses import dataclass
from enum import Enum


class ReadMode(Enum):
    """
    Détermine comment un Subscriber doit lire ce canal. Ce choix est une
    propriété du CANAL, pas de l'appelant — voir Subscriber.receive()
    dans core/zmq_bus.py.

    FIFO : tous les messages sont lus, dans l'ordre, aucune perte.
           (rien dans le pipeline actuel n'en a besoin — réservé pour un
           futur canal d'enregistrement, hors scope ici)

    LIFO : seul le dernier message disponible est retourné ; les messages
           intermédiaires accumulés sont jetés. C'est le mode de tous les
           canaux d'analyse temps réel du pipeline : une donnée périmée
           n'a plus de valeur dès qu'une plus récente existe.
    """
    FIFO = "fifo"
    LIFO = "lifo"


@dataclass(frozen=True)
class ChannelSpec:
    """
    Description statique d'un canal du bus interne.

    Hypothèse d'architecture actée : un seul écrivain (publisher) par canal.
    Si un jour deux process doivent écrire sur le même canal logique, un
    ChannelSpec simple ne suffit plus — il faudrait un broker XSUB/XPUB
    (pattern volontairement non implémenté ici, aucun besoin identifié
    dans le pipeline actuel).
    Si un broker est un jour nécessaire, voir code https://github.com/FPMSEurobot/PAMIRobotPP (commit du 1/05/26)
    """
    name: str            # identifiant humain (logs, erreurs, debug)
    address: str          # ipc:///tmp/... — le Publisher fait le bind() ici
    topic: bytes          # UN SEUL topic par canal (règle actée : pas de
                          # multi-topic sur un canal LIFO, ambigu à la lecture)
    mode: ReadMode        # FIFO ou LIFO — cf ReadMode
    hwm: int = 2           # high-water mark. Bas par défaut : en LIFO ça ne
                          # change pas la donnée retournée (on draine de
                          # toute façon jusqu'au dernier message), ça borne
                          # juste le pire cas de latence/mémoire si personne
                          # ne lit pendant un moment.


# ─────────────────────────────────────────────────────────────────────────
# Chemin 1 — Visibilité des zones : caméra → p_aruco → raycasting → CB zones
# ─────────────────────────────────────────────────────────────────────────

ARUCO_TO_RAYCASTING = ChannelSpec(
    name="aruco_to_raycasting",
    address="tcp://127.0.0.1:5555",
    topic=b"p_aruco.tags",
    mode=ReadMode.LIFO,
)
# Sert à communiquer entre le process aruco_tag_tracking et le process raycasting la position des robots

RAYCASTING_TO_ZONE_STATE = ChannelSpec(
    name="raycasting_to_zone_state",
    address="tcp://127.0.0.1:5556",
    topic=b"zone.state",
    mode=ReadMode.LIFO,
)

# ─────────────────────────────────────────────────────────────────────────
# Chemin 2 — Détection/estimation : caméra → prétraitement → yolo → kf → CB
# ─────────────────────────────────────────────────────────────────────────

YOLO_TO_KALMAN = ChannelSpec(
    name="yolo_to_kalman",
    address="tcp://127.0.0.1:5557",
    topic=b"yolo.detections",
    mode=ReadMode.LIFO,
)

KALMAN_TO_ESTIMATION = ChannelSpec(
    name="kalman_to_estimation",
    address="tcp://127.0.0.1:5558",
    topic=b"kf.estimation",
    mode=ReadMode.LIFO,  # acté dans la spec : postraitement lit LIFO le dernier résultat KF
)

# ─────────────────────────────────────────────────────────────────────────
# Chemin 3 — Homographie (qui est pour l'instant dans l'initialisation) vers Application homographie pour redresser la frame
# ─────────────────────────────────────────────────────────────────────────

INIT_TO_APP_HOMO = ChannelSpec(
    name="init_to_app_homo",
    address="tcp://127.0.0.1:5559",
    topic=b"homographie.matrice",
    mode=ReadMode.LIFO,
)

# Pas besoin je crois
APP_HOMO_TO_INIT = ChannelSpec(
    name="app_homo_to_init",
    address="tcp://127.0.0.1:5560",
    topic=b"kf.estimation",
    mode=ReadMode.LIFO,  # acté dans la spec : postraitement lit LIFO le dernier résultat KF
)