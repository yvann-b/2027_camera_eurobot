"""
Couche runtime du bus ZMQ interne.

Ce fichier répond à "comment on utilise un canal", jamais à "quels canaux
existent" (ça, c'est config/bus_config.py).

Contient :
- la gestion du zmq.Context, un seul par processus OS, recréé automatiquement
  si on détecte un changement de PID (cas d'un fork multiprocessing) — aucun
  appel explicite requis, c'est transparent pour l'appelant.
- Publisher / Subscriber : construits à partir d'un ChannelSpec, picklables
  tels quels (aucune socket n'existe avant le premier .send()/.receive()),
  donc passables directement à mp.Process(args=...).
- Subscriber.receive() applique automatiquement le mode FIFO/LIFO déclaré
  par le ChannelSpec — l'appelant n'a pas à choisir.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional
from dataclasses import is_dataclass, asdict
import zmq

from bus_zmq.bus_config import ChannelSpec, ReadMode


# ── Context : un seul par processus, invalidé et recréé après un fork ──────

_ctx_lock = threading.Lock()
_ctx: Optional[zmq.Context] = None
_ctx_pid: int = -1

def dataclass_encoder(obj):
    if is_dataclass(obj):
        return asdict(obj) # Convertit automatiquement la classe en dictionnaire
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _get_context() -> zmq.Context:
    """
    Ne jamais appeler zmq.Context() directement ailleurs dans le pipeline —
    toujours passer par ici, pour garantir qu'un Context hérité d'un fork
    (donc avec ses threads I/O restés dans le process parent) n'est jamais
    réutilisé tel quel.
    """
    global _ctx, _ctx_pid
    pid = os.getpid()
    with _ctx_lock:
        if _ctx is None or _ctx_pid != pid:
            _ctx = zmq.Context()
            _ctx_pid = pid
    return _ctx


@dataclass
class ReceivedMessage:
    frame_id: int
    timestamp_ns: int
    data: Any


# ── Publisher ────────────────────────────────────────────────────────────

class Publisher:
    """
    Publie sur le canal décrit par `channel`. Un seul Publisher par thread
    (contrainte ZMQ standard : une socket n'est pas thread-safe).
    """

    def __init__(self, channel: ChannelSpec):
        self._channel = channel                             # Picklabel (juste la configuration donc des bits / str / int)
        self._socket: Optional[zmq.Socket] = None           # PAS Picklabel mais donc devra être créer par le processus enfant
        self._pid: int = -1

    def _ensure_socket(self) -> None:
        pid = os.getpid()
        if self._socket is None or self._pid != pid:        # Pour vérifier si on a changé de PID donc de process donc qu'on doit (re)créer le socket
            sock = _get_context().socket(zmq.PUB)           # On récupère le context du process et on crée un socket de type PUB&subscribe
            sock.setsockopt(zmq.SNDHWM, self._channel.hwm)  # On définit la taille max du buffer du socket
            sock.setsockopt(zmq.LINGER, 0)                  # à la fermeture, jette tout ce qui traine et ferme le socket
            sock.bind(self._channel.address)   # le PUB est le point fixe
            self._socket = sock
            self._pid = pid

    def send(self, frame_id: int, timestamp_ns: int, data: Any) -> None:
        """
        frame_id et timestamp_ns sont obligatoires (pas de valeur par
        défaut) : convention non-négociable de la spec, encodée ici dans
        la signature plutôt que laissée à la discipline de chacun.
        """
        self._ensure_socket()
        payload = json.dumps({
            "frame_id": frame_id,
            "timestamp_ns": timestamp_ns,
            "data": data,
        }, default=dataclass_encoder).encode()                                         # On ne peut envoyer que des bits par ZMQ, donc on fait Objet python - dumps > JSON - encode > bits
        self._socket.send_multipart([self._channel.topic, payload])

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    # picklabilité : jamais la socket, seulement le ChannelSpec (donnée pure)
    def __getstate__(self):
        return {"_channel": self._channel}

    def __setstate__(self, state):
        self._channel = state["_channel"]
        self._socket = None
        self._pid = -1

    # Nécessaire parce que Windows et Linux ne travaille pas de la même manière sur l'héritage : sur Linux le fork hérite déjà de None et donc on a pas besoin de setstate pour remettre à 0 lors de la copie / héritage
    # Sur windows pas le cas donc
    # CEPENDANT, si le parent appelle send, il crée le socket, et donc l'enfant hérite du socket. Or le socket n'est PAS picklabel donc ça casse tout. C'est une double sécurité en gros

# ── Subscriber ───────────────────────────────────────────────────────────

class Subscriber:
    """
    S'abonne au canal décrit par `channel`. Le mode de lecture (FIFO/LIFO)
    est entièrement déterminé par channel.mode — receive() ne prend jamais
    ce choix en paramètre, il applique celui du canal.
    """

    def __init__(self, channel: ChannelSpec):
        self._channel = channel
        self._socket: Optional[zmq.Socket] = None
        self._pid: int = -1

    def _ensure_socket(self) -> None:
        pid = os.getpid()
        if self._socket is None or self._pid != pid:
            sock = _get_context().socket(zmq.SUB)
            sock.setsockopt(zmq.RCVHWM, self._channel.hwm)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SUBSCRIBE, self._channel.topic)
            sock.connect(self._channel.address)
            self._socket = sock
            self._pid = pid

    def receive(self, timeout: Optional[float] = None) -> Optional[ReceivedMessage]:
        """
        timeout en secondes. None = bloque indéfiniment. 0 = non-bloquant.
        Retourne None si rien n'est disponible (timeout écoulé, ou canal vide).
        """
        self._ensure_socket()
        if self._channel.mode is ReadMode.FIFO:
            return self._receive_fifo(timeout)
        return self._receive_lifo(timeout)

    def _receive_fifo(self, timeout: Optional[float]) -> Optional[ReceivedMessage]:
        if timeout is not None and not self._socket.poll(int(timeout * 1000), zmq.POLLIN):
            return None
        flags = zmq.NOBLOCK if timeout == 0 else 0
        _, payload = self._socket.recv_multipart(flags=flags)
        return self._decode(payload)

    def _receive_lifo(self, timeout: Optional[float]) -> Optional[ReceivedMessage]:
        # On attend au moins un message si un timeout > 0 est donné...
        if timeout is not None and not self._socket.poll(int(timeout * 1000), zmq.POLLIN):
            return None
        # ...puis on draine tout ce qui est déjà arrivé, on ne garde que le dernier.
        last_payload = None
        while self._socket.poll(0, zmq.POLLIN):
            _, last_payload = self._socket.recv_multipart(flags=zmq.NOBLOCK)
        if last_payload is None:
            return None
        return self._decode(last_payload)

    @staticmethod
    def _decode(payload: bytes) -> ReceivedMessage:
        obj = json.loads(payload)
        return ReceivedMessage(obj["frame_id"], obj["timestamp_ns"], obj["data"])

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def __getstate__(self):
        return {"_channel": self._channel}

    def __setstate__(self, state):
        self._channel = state["_channel"]
        self._socket = None
        self._pid = -1
