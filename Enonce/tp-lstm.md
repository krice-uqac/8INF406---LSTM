<img style="width:25%; height:auto;" src="https://www.uqac.ca/wp-content/themes/uqac/assets/images/uqac.svg" alt="UQAC">>
<h1 style="color:#6b8915">Mise en situation</h1>
<p>Vous êtes un/une spécialiste de la science des données et de l’intelligence artificielle pour le compte de la compagnie d’électricité de la Roumanie.  À l’aide de données historiques, vous devez développer un modèle d’intelligence artificielle qui permettra de prédire la consommation d’électricité et la production d’électricité pour votre entreprise.  Ainsi, vous avez développé un modèle LSTM qui permettra de faire cette prédiction et vous devez établir la grandeur des fenêtres temporelles optimales pour faire cette prédiction.</p>
<h1 style="color:#6b8915">Travail à faire</h1>
<p>Dans le cadre de cette séance, vous devez vous familiariser avec quelques concepts aux niveaux de l’intelligence artificielle.  Vous devez répondre à deux questions théoriques sur les réseaux de neurones en plus de faire quelques expérimentations avec le modèle LSTM fournit en classe.</p>
<li style="padding-left:20px">Question # 1 (4 points)</li>
<p>Faites une recherche sur l’algorithme de rétropropagation du gradient (backpropagation) qui est une pièce fondamentale des réseaux de neurones (1/2 page)</p>
<li style="padding-left:20px">Question # 2 (4 points)</li>
<p>Expliquer brièvement ce que faire un réseau de neurone récurant dans lequel on retrouve les LSTM (1/2 page)<p>
<listyle="padding-left:20px">Question # 3 (2 points)</li>
<p>Faites une analyse sommaire des résultats que vous avez obtenus avec vos expérimentations du modèle fournie en classe. (1/2 pages)</p>
<h1 style="color:#6b8915">Procédure d'installation du code</h1>
<p>Le code LSTM pour l’exécution de votre travail est situé sur GitHub à l’adresse suivante.  Le repo est public et vous y avez accès en suivant l’URL suivant.</p>
<a href=https://github.com/krice-uqac/8INF406---LSTM>https://github.com/krice-uqac/8INF406---LSTM</a>
<br></br>

```
git clone https://github.com/krice-uqac/8INF406---LSTM
```
<p>Ensuite, vous devez créer un environnement virtuel python en utilisant la commande suivante :</p>

```
python -m venv .env[ou autre nom]
```

Ensuite, lance l'environnement avec la commande :

```
.env\Scripts\activate.bat pour Window

ou

source .env/bin/activate pour Linux et MacOS
```
<p>Vous devez installer les dépendances du projet qui sont dans le fichier requirements.txt</p>

```
pip install -–no-cache-directory -r requirements.txt
```

<p>Maintenant, vous devez installer pytorch manuellement</p>

```
Pour GPU, exécuter la commande suivante :

Windows: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
Linux: pip3 install torch torchvision torchaudio

Pour CPU, exécuter la commande suivante :

Windows/Mac : pip3 install torch torchvision torchaudio
Linux : pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
<p>Pour changer des paramètres pour l’exécution du code, il faut le faire dans le fichier config.py</p>

```
DATA_FOLDER='./data'
TIME_WINDOWS_LENGTH = 24*7 # 24 hours * 7 days
OVERLAP = 2 # 6 hours
PREDICTION_LENGTH = 24*7 # 24 hours * 3 days
TRAIN_VALID_TEST_SPLIT = [0.70, 0.15, 0.15] # 70% train, 15% valid, 15% test

# Model parameters
BATCH_SIZE = 32 # batch size for dataloader
LEARNING_RATE = 0.001# number of workers for dataloader
HIDDEN_SIZE = 128 # hidden size of LSTM
NUM_LAYERS = 2 # number of layers of LSTM
EPOCHS = 50
```