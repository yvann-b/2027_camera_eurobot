INCONNU, VIDE, BLEU, JAUNE = -1, 0, 1, 2

_COLOR_TO_INT = {
    "": VIDE,          # EMPTY dans process_post_traitement.py
    "blue": BLEU,
    "yellow": JAUNE,
    "unknown": INCONNU,
}


def _encode_slot(color: str) -> int:
    """Traduit une couleur telle que produite par `output` dans
    process_post_traitement.py ("", "blue", "yellow" ou "unknown") en code
    entier pour le bus robot."""
    return _COLOR_TO_INT.get(color, INCONNU)


def build_robot_payload(
    dico_zones: dict,
    slot_colors_by_zone: dict,  # c'est `output` dans process_post_traitement.py :
                                 # {zone_id: [couleur_slot0, ..., couleur_slot3]}
                                 # ATTENTION : necessite le fix output[idx] (pas output[i])
    frame_id: int,
    timestamp_ns: int,
) -> dict:
    ramassage, depot = [], []

    for zone in dico_zones.values():
        visible = bool(zone.is_visible)
        obstrue = bool(zone.is_blocked)
        fiable = visible and not obstrue  # cf politique de staleness du projet

        if zone.header.type == "ramassage":
            # output marque deja "unknown" les zones invisibles (une fois le
            # bug output[i]/output[idx] corrige) : pas besoin de refaire le
            # check `fiable` ici, on fait juste confiance a `output`.
            colors = slot_colors_by_zone.get(zone.header.id, ["unknown"] * 4)
            ramassage.append({
                "id": zone.header.id,
                "visible": int(visible),
                "obstrue": int(obstrue),
                "slots": [_encode_slot(c) for c in colors],
            })
        elif zone.header.type == "depot":
            depot.append({
                "id": zone.header.id,
                "visible": int(visible),
                "obstrue": int(obstrue),
                # ici pas d'equivalent a `output` : number_blue/number_yellow
                # ne sont jamais remis a "unknown" quand la zone est invisible
                # (seul zone.dominant_color l'est), donc ce garde-fou reste
                # necessaire, contrairement aux slots ramassage ci-dessus.
                "bleu": int(zone.number_blue) if fiable else INCONNU,
                "jaune": int(zone.number_yellow) if fiable else INCONNU,
            })

    return {
        "frame_id": frame_id,
        "timestamp_ns": timestamp_ns,
        "ramassage": ramassage,
        "depot": depot,
    }