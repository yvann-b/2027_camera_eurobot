import zmq
import msgpack


class RobotBusClient:
    def __init__(self, ip, port, topic, enabled=True, sndhwm=5, linger_ms=0):
        self.enabled = enabled
        self.topic = topic
        self._ctx = None
        self._sock = None

        if not self.enabled:
            print("[RobotBus] Desactive par config, aucun socket cree.")
            return

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, sndhwm)
        self._sock.setsockopt(zmq.LINGER, linger_ms)
        self._sock.connect(f"tcp://{ip}:{port}")
        print(f"[RobotBus] Publisher connecte a {ip}:{port}")

        self._send_startup_probe()

    def _send_startup_probe(self):
        # Utile pour valider la connexion, mais attention au "slow joiner" :
        # si le SUB (robot/simulateur) n'est pas encore connecte a cet instant,
        # ce message precis sera perdu silencieusement (comportement normal
        # de PUB/SUB). Ce n'est pas un bug, juste une limite du pattern.
        try:
            self._sock.send_multipart([self.topic, msgpack.packb({"test": True, "msg": "startup"})])
            print("[RobotBus] Message de test envoye")
        except Exception as e:
            print(f"[RobotBus] Erreur envoi message test: {e}")

    def publish(self, payload: dict):
        """Publie payload (dict serialisable msgpack) sur le topic configure.
        No-op silencieux si le bus est desactive (POC / dev sans robot).
        Rappel convention : payload doit porter frame_id et timestamp_ns."""
        if not self.enabled:
            return
        try:
            self._sock.send_multipart([self.topic, msgpack.packb(payload)])
        except Exception as e:
            print(f"[RobotBus] Erreur d'envoi: {e}")

    def close(self):
        if self._sock is not None:
            self._sock.close()
        if self._ctx is not None:
            self._ctx.term()