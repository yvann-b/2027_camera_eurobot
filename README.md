

## Système de Vision Caméra (Année 2026-2027 ???)
Ce projet est le module de vision par ordinateur du robot, conçu autour d'une architecture Python multiprocessus hautement concurrente. Il acquiert les flux vidéo en temps réel, fusionne les détections d'objets (YOLO / ArUco) via un filtre de Kalman, et publie les données spatiales vers la machine d'état principale.

## Installation et Dépendances

Ce projet utilise [uv](https://github.com/astral-sh/uv) pour la gestion ultra-rapide des dépendances et de l'environnement virtuel. Les dépendances sont listées dans le fichier `pyproject.toml`.

Pour installer l'environnement et lancer le projet :
```bash
# Installe les dépendances et crée l'environnement virtuel automatiquement
uv sync

# Lancer le point d'entrée principal
uv run main.py
```

Pour installer la caméra Daheng, il faut : 

-Installer le software GalaxyViewer : https://va-imaging.com/pages/requestdownload (oui il faut créer un compte pour accéder au soft...) 

-J'ai perdu mon fichier tutoriel, donc pour l'utiliser en Python il y a quelques galères en plus, notamment installer iai-gxipy (la librairie non-officielle de GalaxyViewer en Python) et possiblement changer des variables d'environnement, mais j'ai perdu mon doc sur ça, désolé. Dans docs/ se trouve la documentation de iai-gxipy/GalaxyViewer qui peut être très utile. 


