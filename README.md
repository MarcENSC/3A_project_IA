# Projet de Reinforcement Learning - ENSC 2025

## Description du Projet
Ce projet s'inscrit dans le cadre du projet de fin d'études de spécialisation en Intelligence Artificielle à l'École Nationale Supérieure de Cognitique (ENSC), promotion 2025.

L'objectif est d'entraîner un agent en utilisant différents algorithmes d'apprentissage par renforcement :
- D SARSA (State-Action-Reward-State-Action)
- DQN (Deep Q-Network)

## Prérequis
- Python 3.8
- Conda (recommandé)

## Installation et Configuration

0. Cloner le répo Github
```bash
git clone https://github.com/MarcENSC/3A_project_IA.git
cd 3A_project_IA
```
### Avec Conda (Recommandé)
1. Créer un nouvel environnement conda avec Python 3.8 :
```bash
conda create --name projet_IA python=3.8 pip --channel conda-forge
```

2. Activer l'environnement :
```bash
conda activate projet_IA
```

3. Installer les dépendances requises :
```bash
pip install -r requirements.txt
```



## Exécution du projet

### Entraînement du modèle
Pour lancer l'entraînement du modèle :
```bash
python src/main.py
```

### Test du modèle
Pour tester le modèle entraîné :
```bash
python src/test.py
```

⚠️ **Warning:** Pour voir ou ne pas voir visuelllement la progression de l'agent Mario dans l'environnement, il est nécessaire de modifier l'argument dans le fichier **wrapper.py**  à la ligne **34** :  **render_mode** et changer les valeurs entre **rgb_array** ou **human** 


## Structure du Projet

```
3A_PROJECT_IA/
├── src/
│   ├── agent/                      # Implémentation des agents RL
│   │   ├── dqn_agent.py           # Agent utilisant DQN
│   │   └── sarsa_agent.py         # Agent utilisant SARSA
│   │
│   ├── env_wrappers/              # Wrappers pour l'environnement
│   │   ├── skip_frame_wrapper.py  # Gestion de différents wrappers
│   │   └── wrapper.py             # wrapping de l'environnement 
│   │
│   ├── logs/                      # Gestion des logs
│   │   └── logger.py              # Configuration des logs
│   │
│   └── network/                   # Architecture réseau
│       └── dqn_network.py         # Réseau de neurones pour DQN et SARSA
│
├── main.py                        # Script principal d'entraînement
├── test.py                        # Script de test des modèles
├── dqn_model_weights.chkpt        # Poids du modèle entraîné
└── requirements.txt               # Dépendances du projet
```

## Relations entre les composants

### Agents (src/agent/)
- `dqn_agent.py` : Implémente l'agent DQN qui utilise le réseau de neurones défini dans `network/dqn_network.py`
- `sarsa_agent.py` : Implémente l'agent SARSA avec sa propre logique d'apprentissage

### Environment Wrappers (src/env_wrappers/)
- `wrapper.py` : Wrapper principal qui prépare l'environnement pour l'apprentissage
- `skip_frame_wrapper.py` : gestion des différents wrappers 

### Network (src/network/)
- `dqn_network.py` : Définit l'architecture du réseau de neurones utilisé par l'agent DQN

### Scripts Principaux
- `main.py` : script  principal qui :
  - Initialise l'environnement avec les wrappers appropriés
  - Crée l'agent (DQN ou SARSA)
  - Lance l'entraînement
  - Sauvegarde les poids du modèle

- `test.py` : Script d'évaluation qui :
  - Charge un modèle entraîné
  - Exécute un épisode de test
  - Affiche les performances



