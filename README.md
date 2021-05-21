# statapp

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/baptiste-pasquier/statapp/tree/master/)

Construction d’un modèle de prédiction de clic sur des publicités en ligne avec contraintes sur la quantité de données et le temps d’évaluation.

[Note de synthèse](Synthèse.pdf)

[Rapport](Rapport.pdf)

-----------------

## Installation

* Installation de l'environnement Anaconda

Cloner le répertoire puis exécuter à l'intérieur du répertoire : 
```bash
conda env create
```

Instructions spécifiques à JupyterLab pour l'affichage des ProgressBar tqdm et des graphiques interactifs:
```bash
conda activate statapp
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib
```

* Lancement de JupyterLab ou Jupyter Notebook

Exécuter à l'intérieur du répertoire : 
```bash
conda activate statapp
jupyter lab
```
ou : 
```bash
conda activate statapp
jupyter notebook
```

-----------------

2020-2021
