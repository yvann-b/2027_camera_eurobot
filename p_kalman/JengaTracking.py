import numpy as np

class EstimatedObject :
    """
    Convetion :
    x_k_k1 = x à l'état k ayant des informations jusqu'à l'état k-1
    Modèle à position constante :
    X_k_k1 = X_k_k
    """

    def __init__(self, couleur, x, y, theta_r, initial_covariance,measurement_incertitude,model_incertitude):
        self.couleur = couleur
        self.estimated_state = np.array([x, y, theta_r], dtype=float)
        self.estimated_error_covariance = initial_covariance
        self.measurement_incertitude = measurement_incertitude
        self.model_incertitude = model_incertitude
        self.number_nohits = 0

        # ======= Matrice =======
        '''
        Vecteur d'état : [x y theta_r]^T
        modèle : [x y theta_r]^T |k = [x y theta_r]^T |k-1 + bruit -> A = I
        mesure : z = [~x ~Y ~theta_r] |k = [x y theta_r]^T |k + bruit -> C = I 
        '''
        self.A = np.eye(3)
        self.C = np.eye(3)


        # ======= Pour regrouper les attributs de la class
        self.projected_state = None
        self.projected_error_covariance = None
        self.kalman_gain = None

    def __repr__(self):
        return f"Obj : color : {self.couleur} | xyt : {self.estimated_state}"

    def get_xy(self):
        return self.estimated_state[0], self.estimated_state[1]

    def get_augmented_state(self):
        dico = {"x_mm" : self.estimated_state[0], "y_mm" : self.estimated_state[1], "theta_r" : self.estimated_state[2], "color" : self.couleur}
        return dico

    def predict(self):
        self.projected_state = self.estimated_state
        self.projected_error_covariance = self.A @ self.estimated_error_covariance @ self.A.T + self.model_incertitude


    def update(self, measurement):
        if measurement is None:
            self.number_nohits += 1
            if self.number_nohits > 10:
                return False            # If the object is not detected for more than 10 frames -> Send the signals to kill itself
            self.estimated_state = self.projected_state                         # No data -> Predict becomes estimation
            self.estimated_error_covariance = self.projected_error_covariance   # No data -> Predict becomes estimation
            return True                 # Else return true
        if self.couleur != measurement["class"] :
            return False                                                        # Two differents objects

        z = np.array([measurement["x_mm"], measurement["y_mm"], measurement["theta_rad"]])
        # z est sous la forme attendue pour le filtrage de Kalman

        # Computation of Kalman Gain
        self.kalman_gain = self.projected_error_covariance @ self.C @ np.linalg.inv(self.C @ self.projected_error_covariance @ self.C.T + self.measurement_incertitude)

        # Update of state variable
        # As we use angle (radian) we have to make sure that we use "the absolute" difference
        innovation = z - self.C @ self.projected_state
        innovation[2] = (innovation[2] + np.pi) % (2 * np.pi) - np.pi           # Error bring between -pi and pi

        self.estimated_state = self.projected_state + self.kalman_gain @ innovation

        # update of the error covariance
        self.estimated_error_covariance = (np.eye(3) - self.kalman_gain @ self.C) @ self.projected_error_covariance

        # Reset of number of hits :
        self.number_nohits = 0

        return True