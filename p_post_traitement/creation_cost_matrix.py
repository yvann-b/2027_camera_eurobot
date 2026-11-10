import numpy as np

def creation_cost_matrix(liste_slot, dico_measurement):

    estimations = dico_measurement["data"]
    number_estimation = len(estimations)
    number_slot = len(liste_slot)

    cost_mtx = np.zeros((number_estimation,number_slot))

    for i,est in enumerate(estimations) :
        for j,slot in enumerate(liste_slot) :
            x_e = est[0]
            y_e = est[1]
            x_s = slot[0]
            y_s = slot[1]
            distance = np.sqrt((x_e - x_s)**2 + (y_e - y_s)**2)
            cost_mtx[i, j] = distance
    return cost_mtx