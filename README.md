# Projet de Renforcement Learning - ENSC 2025

## Description du Projet
Ce projet s'inscrit dans le cadre du projet de fin d'études de spécialisation en Intelligence Artificielle à l'École Nationale Supérieure de Cognitique (ENSC), promotion 2025.

L'objectif est d'entraîner un agent en utilisant différents algorithmes d'apprentissage par renforcement :
- D SARSA (State-Action-Reward-State-Action)
- DQN (Deep Q-Network)

## Prérequis
- Python 3.8
- Conda (recommandé)

## Installation et Configuration

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
- `skip_frame_wrapper.py` : Optimise l'apprentissage en permettant de sauter des frames

### Network (src/network/)
- `dqn_network.py` : Définit l'architecture du réseau de neurones utilisé par l'agent DQN

### Scripts Principaux
- `main.py` : Orchestrateur principal qui :
  - Initialise l'environnement avec les wrappers appropriés
  - Crée l'agent (DQN ou SARSA)
  - Lance l'entraînement
  - Sauvegarde les poids du modèle

- `test.py` : Script d'évaluation qui :
  - Charge un modèle entraîné
  - Exécute des épisodes de test
  - Affiche les performances

## Exécution du projet

### Entraînement du modèle
Pour lancer l'entraînement du modèle :
```bash
python main.py
```

### Test du modèle
Pour tester le modèle entraîné :
```bash
python test.py
```
