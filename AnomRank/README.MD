
#Comment lancer AnomRank ? 

- Ouvrir le fichier run.sh : 
    Il est constitué de 2 lignes : 
      "g++ -std=c++11 -stdlib=libc++ -O3 main.cpp -o anomrank;
       ./anomrank "darpa.txt" " " 60 256 0 50 70"
       
- Ouvrir un terminal de commande 
- Aller dans le dossier AnomRank
- Executer la première ligne du fichier run.sh
- Executer la deuxième ligne du fichier run.sh

- En sortie, un fichier .txt a été créé pour regrouper les résultats

#Description de la deuxième ligne du fichier run.sh : 

Rappel : "./anomrank "darpa.txt" " " 60 256 0 50 70"

- ./anomrank permet d'executer la fonction "main"" du fichier "main.cpp" 

7 arguments sont demandés enuite : 
Ils correspondent respectivement aux paramètres au début de la fonction "main" 
du fichier "main.cpp" : 

    string path = argv[1];
    string delimeter = argv[2];
    int timeStep = atoi(argv[3]);
    int initSS = atoi(argv[4]);
    int injectScene = atoi(argv[5]);
    int injectNum = atoi(argv[6]);
    int injectSize = atoi(argv[7]);
    
- path : nom du dataset utilisé pour AnomRank 
- delimiter : delimiter utilisé entre les variables du datasets; 
  en l'occurrence un espace. 
- timeStep : timeStep utilisé (voir un peu plus loin)
- initSS : (voir un peu plus loin)
- injectScene : Dans le code d'anomRank, on a la possibilité d'ajouter des 
  anomalies en modifiant le dataset. Si injectScene != 0 alors on souhaite 
  ajouter des anomalies.
- injectNum : Nombre d'anomalies à ajouter
- injectSize : Taille des anomalies à ajouter

Par exemple : ./anomrank "darpa.txt" " " 60 256 0 50 70
Cela signifie qu'on applique anomrank sur le fichier darpa.txt, 
dont les variables sont délimitées par un espace (" "),
on utilise un timeStpe de 60,
un initSS de 256,
et on ne souhaite pas ajouter d'anomalies (0),
donc les valeurs 50 et 70 sont choisies au hasard.


# Revenons sur timeStep et initSS : 

Ces paramètres permettent de définir le nombre de données qui vont être testées 
par anomRank.
Cela est représenté par : int testNum = numSS - initSS; (toujours dans main.cpp)

Avec :
  int numSS = timeNum/timeStep + 1; 
  où timeNum correspond au nombre de timesatmp du dataset.
  


# Quels sont les fichiers ?

- accuracy.cpp : il calcule la précision de l'algorithme (precision, recall, 
  false positive etc.)

- anomaly_detect.cpp : il est notamment utilisé pour calculer le score d'anomalie
  d'un instant avec la fonction "compute_anomaly_score"
  
- anomaly_inject.cpp : il est utilisé pour ajouter des anomalies dans le dataset

- main.cpp : fichier principal permettant d'exécuter anomarank sur le dataset

- pagerank.cpp : Il permet de calculer le pagerank (version Structure et version
  Weight des instants), utilisé ensuite pour le calcul du score d'anomalie
  
- read_data.cpp : Il permet de lire les données du dataset pour ensuite 
  pouvoir travailler dessus







