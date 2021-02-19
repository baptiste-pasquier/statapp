# statapp


## Table des matières
* [A. Installation](#A-Installation)

## A. Installation

* Installation de l'environnement d'exécution avec Anaconda

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
