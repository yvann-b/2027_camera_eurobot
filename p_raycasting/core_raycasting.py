import numpy as np

def intersection_droite_cylindre(center, point_table,point_camera,id = "" ,height = 430, radius = 250):
    """
    :param center:              Position du robot (xyz)
    :param point_table:         Position du centre de la zone
    :param point_camera:        Position dans le repère de la table de la caméra (t_cam)
    :param id:                  Pour du print / débug
    :param height:              Hauteur max du robot (considéré ici comme un cylindre)
    :param radius:              Rayon du robot
    :return:                    True : intersection ; False : pas intersection
    """

    ux = point_camera[0][0] - point_table[0]
    uy = point_camera[1][0] - point_table[1]
    uz = point_camera[2][0] - point_table[2]
    # Le vecteur directeur n'a pas besoin d'être normalisé


    temp1 = point_table[0] - center[0]
    temp2 = point_table[1] - center[1]

    A = (ux**2 + uy**2)
    B = (2*ux*temp1 + 2*uy*temp2)
    C = temp1**2 + temp2**2 - radius**2

    delta = B**2 - 4*A*C
    if delta >= 0:
        lambda1 = (-B+np.sqrt(delta))/(2*A)
        lambda2 = (-B - np.sqrt(delta)) / (2 * A)
        if 0 <lambda1 <1 or 0 <lambda2 <1:
            P1z = lambda1 * uz + point_table[2]              # Point table de 2 vaudra toujours 0 mais bon au moins je suis rigoureux
            P2z = lambda2 * uz + point_table[2]              # Point table de 2 vaudra toujours 0 mais bon au moins je suis rigoureux
            if 0 < P1z < height or 0 < P2z < height:
#                print(f"Hauteur détection : {P1z} et {P2z} pour la zone : {id}")
                return True

    return False