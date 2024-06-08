# Floutage automatique de vidéo à partir d'un modèle YOLOv8

Ce projet permets de flouter automatiquement une vidéo contenue dans un fichier .bag (au format ROS1 [ROS2 pas encore implémenté]).

## Prérequis
Le projet a été développé sous Python 3.10, donc il est conseillé d'avoir une version supérieure ou égale à celle-ci (aucun test n'a été fait avec une version antérieure).

## Installation
D'abord, il faut cloner ce repo Git, en faisant :
```
git clone https://github.com/franzele21/pji.git
```
Puis on rentre dans le projet, et on télécharge les modules nécessaires :
```
cd pji
python3 -m pip install -r requirements.txt
```

## Utilisation
Pour utiliser le programme, il suffit, dans la ligne de commande, d'écrire la commande suivante :
```
python3 blur_bag.py chemin/vers/fichier.bag chemin/vers/modèle.pt
```

Avec le premier argument étant le chemin vers le fichier bag possédant la vidéo à flouter, et le deuxième argument est le modèle Yolov8 qui servira à trouver les zones à flouter (par exemple, on mettra un modèle qui peut trouver les [visages](https://github.com/akanametov/yolov8-face) pour les flouter).

On peut aussi rajouter des options suivantes :
| Commande          | Autre forme   | Description |
|-------------------|---------------|-------------|
| `--output_file`   | `-o`          | Chemin/nom de la vidéo de sortie |
| `--frame_rate`    |               | Intervalle d'échantillonnage |
| `--black_box`     |               | Remplace le floutage par une boite noire (plus rapide que le floutage) |
| `--keep_orig_mp4` |               | Garde la reconstruction du fichier mp4 du fichier bag avant le floutage |
| `--verbose`       | `-v`          | Affiche l'avancement du processus |

## Explication du programme 
Voici comment se déroule le programme :
1. D'abord, on lit le fichier .bag, et on extrait la vidéo. On enregistre cette vidéo dans un fichier mp4 (fonction `bag_to_mp4()`).
2. Dans la fonction `blur_video()`, on appelle la fonction `tmp_video()`, qui créer une vidéo temporaire de la vidéo extraite du fichier bag, mais cette nouvelle vidéo sera plus courte que l'originale : cette fonction va échantillonner chaque $x$ frame. De base, on échantillonne chaque 5 frames de la vidéo du fichier bag, mais on peut changer le nombre en spécifiant l'intervalle d'échantillonage avec l'option de commande `--frame_rate`.
3. De retour dans le contexte de `blur_video()`, on va ouvrir la vidéo originale et la vidéo d'échantillon en parrallèle, et on va parcourir les deux en même temps. Pour chaque frame de la vidéo d'échantillon, on va passer la frame dans le modèle Yolov8, afin de savoir s'il y'a des zones à flouter. S'il n'y a pas de zones à flouter, on enregistre les frames voisines à la frame échantillon sans les vérifier par le modèle. Si on trouve au moins une zone à flouter dans la frame d'échantillon, on va passer les frames voisines de la vidéo originale au modèle, afin de flouter aussi les frames voisines si besoin. L'intervalle d'échantillonnage est de 5 frame de base, mais peut être modifié avec l'option `--frame_rate`.
4. Quand une frame à besoin d'être floutée, on extrait les `box` des zones à flouter, et on les donnent à la fonction `blur_box()`. Si on a mit `--black_box`, on ne va pas flouter les zones, mais on va poser une boîte noir. Cette option est plus rapide que le floutage. 
5. Une fois toute la vidéo faite, on enregistre la vidéo sous le nom `ouput.mp4` ou un nom donné avec l'option `--output_file`
6. On supprime la vidéo d'échantillon, et si l'option `--keep_orig_mp4` n'a pas été passé, on ne garde pas la vidéo extraite du fichier bag.
