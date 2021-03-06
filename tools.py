import os
import pickle
import time
from math import ceil

import shutil
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline
from IPython.display import display
from pandas.api.types import is_numeric_dtype
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer, ConfusionMatrixDisplay
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     RandomizedSearchCV, train_test_split)
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


COLUMNS_QUANT = ['contextid',
                 'campaignctrlast24h',
                 'dayssincelastvisitdouble',
                 'ltf_nbglobaldisplay_4w',
                 'ltf_nbpartnerdisplayssincelastclick',
                 'ltf_nbpartnerdisplay_90d',
                 'ltf_nbpartnerclick_90d',
                 'ltf_nbpartnersales_90d',
                 'nbdayssincelastclick',
                 'nbdisplay_1hour',
                 'nbdisplayglobalapprox_1d_sum_xdevice',
                 'display_size',
                 'zonecostineuro']

COLUMNS_CAT = ['display_env',
               'target_env',
               'campaignscenario',
               'campaignvertical',
               'is_interstitial',
               'device_type',
               'hour',
               'weekday']


# ---------------------------------------------------------------------------- #
#                             Création des datasets                            #
# ---------------------------------------------------------------------------- #

def datasets(df, columns_quant=COLUMNS_QUANT, columns_cat=COLUMNS_CAT, verbose=True, sparse=False, drop='if_binary'):
    if verbose:
        print(f"Columns_quant : {'default' if columns_quant == COLUMNS_QUANT else columns_quant}")
        print(f"Columns_cat : {'default' if columns_cat == COLUMNS_CAT else columns_cat}")
        print(f"drop : {drop}")

    df = df[COLUMNS_QUANT + COLUMNS_CAT + ['is_display_clicked']]
    df = df.dropna()

    X_quant = df[columns_quant]
    X_quant_scaled = (X_quant - X_quant.mean()) / X_quant.std()
    if verbose:
        print(f"\nNombre de variables pour X_quant : {len(X_quant.columns)}\n")
        display(X_quant.columns)

    is_columns_cat_dummy = False    # True si columns_cat contient des dummy variables
    for elem in columns_cat:
        if elem not in COLUMNS_CAT:
            is_columns_cat_dummy = True

    if not is_columns_cat_dummy:
        X_cat = df[columns_cat]
        index = X_cat.index
        enc = OneHotEncoder(drop=drop, sparse=False)
        X_cat = enc.fit_transform(X_cat)
        X_cat = pd.DataFrame(X_cat, columns=enc.get_feature_names(columns_cat), index=index)
    else:
        X_cat = df[COLUMNS_CAT]
        index = X_cat.index
        enc = OneHotEncoder(drop=drop, sparse=False)
        X_cat = enc.fit_transform(X_cat)
        X_cat = pd.DataFrame(X_cat, columns=enc.get_feature_names(COLUMNS_CAT), index=index)
        X_cat = X_cat[columns_cat]

    X_cat_scaled = (X_cat - X_cat.mean()) / X_cat.std()
    X_cat_scaled2 = X_cat / X_cat.std()
    if verbose:
        print(f"\nNombre de variables pour X_cat : {len(X_cat.columns)}\n")
        display(X_cat.columns)

    X = pd.concat([X_quant, X_cat], axis=1)
    X_all_scaled = pd.concat([X_quant_scaled, X_cat_scaled], axis=1)
    X_only_quant_scaled = pd.concat([X_quant_scaled, X_cat_scaled2], axis=1)
    if verbose:
        print(f"\nNombre de variables pour X : {len(X.columns)}")

    y = df['is_display_clicked']

    if sparse:
        dico = {'X_quant': csr_matrix(X_quant),
                'X_quant_scaled': csr_matrix(X_quant_scaled),
                'X_cat': csr_matrix(X_cat),
                'X_cat_scaled': csr_matrix(X_cat_scaled),
                'X': csr_matrix(X),
                'X_only_quant_scaled': csr_matrix(X_only_quant_scaled),
                'X_all_scaled': csr_matrix(X_all_scaled),
                'y': y}
    else:
        dico = {'X_quant': X_quant,
                'X_quant_scaled': X_quant_scaled,
                'X_cat': X_cat,
                'X_cat_scaled': X_cat_scaled,
                'X': X,
                'X_only_quant_scaled': X_only_quant_scaled,
                'X_all_scaled': X_all_scaled,
                'y': y}

    return dico


TIME_INTERVALS = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('h', 3600),    # 60 * 60
    ('min', 60),
    ('s', 1))


def display_time(seconds, granularity=5):
    result = []
    for name, count in TIME_INTERVALS:
        value = int(seconds // count)
        if name == 's':
            value = round(seconds, 3)
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{}{}".format(value, name))
    return ', '.join(result[:granularity])


# ---------------------------------------------------------------------------- #
#                                 Modélisation                                 #
# ---------------------------------------------------------------------------- #

class Modelisation():
    def __init__(self, X, y, model, X_test=None, y_test=None, seuil=None):
        """
        Par défaut : division du dataset (X, y) en un training set et un test set, sauf si (X_test, y_test) est fourni.
        """
        if X_test is not None or y_test is not None:
            assert X_test is not None
            assert y_test is not None

        if X_test is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=1234)
            X_train = pd.DataFrame(data=X_train, columns=X.columns)
            X_test = pd.DataFrame(data=X_test, columns=X.columns)
        else:
            X_train = X
            y_train = y

        # if scaling:
        #     columns = X.columns
        #     scaler = StandardScaler()
        #     X_train = scaler.fit_transform(X_train)
        #     X_test = scaler.transform(X_test)
        #     X_train = pd.DataFrame(data=X_train, columns=columns)
        #     X_test = pd.DataFrame(data=X_test, columns=columns)

        t1 = time.time()
        model.fit(X_train, y_train)
        self.training_time = time.time() - t1
        t1 = time.time()
        if seuil is None:
            y_pred = model.predict(X_test)
        else:
            y_pred = (model.predict_proba(X_test)[:,1] >= seuil).astype(bool)
        self.prediction_time = time.time() - t1

        cm = confusion_matrix(y_test, y_pred)
        probs = model.predict_proba(X_test)[:, 1]

        TP = cm[1][1]
        FN = cm[1][0]
        FP = cm[0][1]
        TN = cm[0][0]
        sc_roc_auc = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # Recall
        Recall = TP / (TP + FN)
        # Precision
        Precision = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # F1_Score
        F1 = (2 * Precision * Recall) / (Precision + Recall)
        # F3_Score
        F3 = (10 * Precision * Recall) / (9 * Precision + Recall)
        # F5_Score
        F5 = (26 * Precision * Recall) / (25 * Precision + Recall)

        metrics_score = {'f1': F1, 'f3': F3, 'f5': F5, 'recall': Recall, 'negative predictive value': NPV, 'precision': Precision, 'roc_auc': sc_roc_auc}

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.model = model
        self.probs = probs
        self.metrics_score = metrics_score
        self.recall = Recall
        self.X_columns = X.columns
        self.seuil = seuil

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def show_conf_matrix(self, pdf=None):
        if self.seuil is None:
            y_pred = self.model.predict(self.X_test)
        else:
            y_pred = (self.model.predict_proba(self.X_test)[:,1] >= self.seuil).astype(bool)
        cm = confusion_matrix(self.y_test, y_pred, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.model.classes_)
        disp.plot(cmap='Blues') 
        if pdf:
            pdf.export()
        plt.show()

    def show_metrics_score(self):
        for key, value in self.metrics_score.items():
            print(f"{key} : {value:.4f}")
        print(f"training time : {display_time(self.training_time)}")
        print(f"prediction time : {display_time(self.prediction_time)}")

    def show_ROC(self, pdf=None):
        fpr, tpr, _ = metrics.roc_curve(self.y_test, self.probs)
        plt.plot(fpr, tpr, label=f"{self.model}")
        plt.plot([0, 1], [0, 1], "r-", label='Modèle aléatoire')
        plt.plot([0, 0, 1], [0, 1, 1], 'b-', label='Modèle parfait')
        plt.legend()
        if pdf:
            pdf.export()
        plt.title('Courbe ROC')
        plt.show()

    def show_attributes(self):
        if isinstance(self.model, sklearn.tree.DecisionTreeClassifier):
            # help(sklearn.tree._tree.Tree)
            tree = self.model.tree_
            attributes = {'max_depth': tree.max_depth, 'n_leaves': tree.n_leaves, 'node_count': tree.node_count}
            for key, value in attributes.items():
                print(f"{key} : {value}")

    # Spécifiques à DecisionTreeClassifier
    def plot_tree(self):
        assert(isinstance(self.model, sklearn.tree.DecisionTreeClassifier))
        dot_data = sklearn.tree.export_graphviz(self.model,
                                                out_file=None,
                                                feature_names=self.X_columns,
                                                class_names=['False', 'True'],
                                                filled=True, rounded=True,
                                                special_characters=True)
        graph = graphviz.Source(dot_data)
        display(graph)

    # Spécifiques à XGBClassifier
    def plot_importance(self, **kwargs):
        assert(isinstance(self.model, xgb.XGBClassifier))
        xgb.plot_importance(self.model, **kwargs)

    def show_graph(self, **kwargs):
        assert(isinstance(self.model, xgb.XGBClassifier))
        display(xgb.to_graphviz(self.model, **kwargs))


# ---------------------------------------------------------------------------- #
#                             Randomized/GridSearch                            #
# ---------------------------------------------------------------------------- #

def f_model_name(model):
    if isinstance(model, XGBClassifier):
        return 'XGBoost'
    elif isinstance(model, LogisticRegression):
        return 'LR'
    elif isinstance(model, DecisionTreeClassifier):
        return 'Tree'
    elif isinstance(model, RandomForestClassifier):
        return 'Forest'
    elif isinstance(model, RandomOverSampler):
        return 'RandomOver'
    elif isinstance(model, SMOTENC):
        return 'SMOTENC'
    elif isinstance(model, Pipeline):
        steps = model.steps
        return f_model_name(steps)
    elif isinstance(model, list):
        if len(model) == 1:
            return f_model_name(model[0][1])
        else:
            return f_model_name(model[0][1]) + '_' + f_model_name(model[1:])
    else:
        return str(model)


SCORING = {'recall': 'recall',
           'precision': 'precision',
           'f1': 'f1',
           'f3': make_scorer(fbeta_score, beta=3),
           'f5': make_scorer(fbeta_score, beta=5)
           }


def SearchCV(model, params, **kwargs):
    # Récupération des arguments
    columns_quant = kwargs.pop('columns_quant', COLUMNS_QUANT)
    columns_cat = kwargs.pop('columns_cat', COLUMNS_CAT)
    drop = kwargs.pop('drop', 'if_binary')
    data_frac = kwargs.pop('data_frac', 1)
    scaling = kwargs.pop('scaling', False)
    sparse = kwargs.pop('sparse', False)

    scoring = kwargs.pop('scoring', SCORING)

    random = kwargs.pop('random', False)
    if random:
        n_iter = kwargs.pop('n_iter', 10)
    random_state = kwargs.pop('random_state', None)

    n_jobs = kwargs.pop('n_jobs', -1)
    name = kwargs.pop('name', '')

    if len(kwargs) > 0:
        raise ValueError(f"Arguments non valides : {[*kwargs]}")

    model_name = f_model_name(model)
    if name:
        model_name += f'_{name}'

    print('RandomizedSearchCV' if random else 'GridSearchCV')
    print(f"Modèle : {model_name}")
    print('******************')
    print(f"\nNombre total de combinaisons de paramètres : {len(ParameterGrid(params))}")
    if random:
        print(f"Nombre de combinaisons aléatoires testées : {n_iter}\n")
    print(f"Columns_quant : {'default' if columns_quant == COLUMNS_QUANT else columns_quant}")
    print(f"Columns_cat : {'default' if columns_cat == COLUMNS_CAT else columns_cat}")
    print(f"drop : {drop}")
    print(f"Pourcentage des données : {data_frac*100}%")

    # Création des datasets
    csv = 'data/df_train_prepro.csv'
    df = pd.read_csv(csv).sample(frac=data_frac, random_state=random_state)
    datasets_df = datasets(df, columns_quant=columns_quant, columns_cat=columns_cat, verbose=False, sparse=sparse, drop=drop)

    if scaling:
        X = datasets_df['X_only_quant_scaled']
    else:
        X = datasets_df['X']
    y = datasets_df['y']

    if random:
        search = RandomizedSearchCV(model, params, n_iter=n_iter, scoring=scoring, refit=False, n_jobs=n_jobs, cv=5, random_state=random_state)
    else:
        search = GridSearchCV(model, params, scoring=scoring, refit=False, n_jobs=n_jobs, cv=5)

    t1 = time.time()
    search.fit(X, y)
    temps = display_time(time.time() - t1)

    results = pd.DataFrame(search.cv_results_)
    results = results.convert_dtypes()

    len_grid = len(ParameterGrid(params))

    dico = {'model': str(model),
            'model_name': model_name,
            'type': 'RandomizedSearchCV' if random else 'GridSearchCV',
            'len_grid': len_grid,
            'n_iter': n_iter if random else '',
            'columns_quant': 'default' if columns_quant == COLUMNS_QUANT else columns_quant,
            'columns_cat': 'default' if columns_cat == COLUMNS_CAT else columns_cat,
            'drop': drop,
            'data_frac': data_frac,
            'n_jobs': n_jobs,
            'temps': temps,
            'params': params,
            'scoring': scoring
            }
    if not random:
        del dico['n_iter']

    filename = f'{model_name}_CV_'
    if random:
        filename += f'Randomized{n_iter}_'
    else:
        filename += 'Grid_'
    filename += f'{len_grid}_'
    filename += f'{data_frac}'

    print(f"\nTemps : {temps}")
    print(f"Exportation : {filename}")

    pickle.dump((dico, results), open('backups/' + filename + '.pkl', 'wb'))


def restauration_CV(filename, verbose=True):
    dico, results = pickle.load(open('backups/' + filename + '.pkl', 'rb'))

    if verbose:
        for key, value in dico.items():
            print(f"{key} : {value}")

    # On enlève toutes les colonnes split
    results = results.loc[:, ~results.columns.str.startswith('split')]

    return dico, results


def graph_2scores_CV(dico, results, score1, score2, s=20, zoom=1, pdf=None):
    """
    Zoom sur les x% meilleurs combinaisons selon score1
    """
    plt.figure(figsize=(14, 8))

    if dico['type'] == 'RandomizedSearchCV':
        n = int(zoom * dico['n_iter'])
    else:
        n = int(zoom * dico['len_grid'])

    results_sort = results.sort_values(by=f'mean_test_{score1}', ascending=False)
    plt.scatter(results_sort[f'mean_test_{score1}'][:n], results_sort[f'mean_test_{score2}'][:n], marker='o', s=s)

    plt.xlabel(score1)
    plt.ylabel(score2)
    if pdf:
        pdf.export()
    if dico['type'] == 'RandomizedSearchCV':
        plt.title(f"{dico['model_name']} | RandomizedSearchCV : {'(zoom) ' if zoom != 1 else ''}scores de {n} combinaisons de paramètres parmi {dico['len_grid']}, avec {dico['data_frac']*100}% des données")
    else:
        plt.title(f"{dico['model_name']} | GridSearchCV : {'(zoom) ' if zoom != 1 else ''}scores de {n} combinaisons de paramètres, avec {dico['data_frac']*100}% des données")
    plt.show()


def graph_3scores_CV(dico, results, score1, score2, score3, s=20, zoom=1, pdf=None):
    """
    Zoom sur les x% meilleurs combinaisons selon score1
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if dico['type'] == 'RandomizedSearchCV':
        n = int(zoom * dico['n_iter'])
    else:
        n = int(zoom * dico['len_grid'])

    results_sort = results.sort_values(by=f'mean_test_{score1}', ascending=False)

    ax.scatter(results_sort[f'mean_test_{score1}'][:n], results_sort[f'mean_test_{score2}'][:n], results_sort[f'mean_test_{score3}'][:n], s=s, color='r', linestyle="None", marker='o')

    ax.set_xlabel(score1)
    ax.set_ylabel(score2)
    ax.set_zlabel(score3)
    if pdf:
        pdf.export()
    if dico['type'] == 'RandomizedSearchCV':
        plt.title(f"{dico['model_name']} | RandomizedSearchCV : {'(zoom) ' if zoom != 1 else ''}scores de {n} combinaisons de paramètres parmi {dico['len_grid']}, avec {dico['data_frac']*100}% des données")
    else:
        plt.title(f"{dico['model_name']} | GridSearchCV : {'(zoom) ' if zoom != 1 else ''}scores de {n} combinaisons de paramètres, avec {dico['data_frac']*100}% des données")
    plt.show()


def graph_param_CV(dico, results, param=None, ncols=3, xscale={}, height=3, width=5, pdf=None):
    """
    xscale = {param1: 'log'}
    """
    if param is None:
        list_param = dico['params'].keys()
    else:
        list_param = [param]

    if len(list_param) > 1:
        ncols = ncols
        nrows = ceil(len(list_param) / ncols)
    else:
        ncols, nrows = 1, 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width * ncols, height * nrows))

    for i, param in enumerate(list_param):
        if len(list_param) == 1:
            ax = axes
        elif len(list_param) <= ncols:
            ax = axes[i % ncols]
        else:
            ax = axes[i // ncols, i % ncols]

        if is_numeric_dtype(results[f'param_{param}']):
            results[f'param_{param}'].fillna(value=-1, inplace=True)
            nb_param = results[f'param_{param}'].nunique()
            numeric = True
        else:
            results[f'param_{param}'] = results[f'param_{param}'].astype(str)
            nb_param = results[f'param_{param}'].nunique()
            numeric = False

        if nb_param <= 15 or not numeric:  # xticks régulier
            r = list(range(nb_param))
            for score in dico['scoring']:
                a = results.groupby(f'param_{param}').mean()
                if param in ['class_weight', 'model__class_weight']:
                    a.sort_index(inplace=True, ascending=True, key=key_class_weight)
                else:
                    a.sort_index(inplace=True, ascending=True)
                ax.plot(r, list(a[f"mean_test_{score}"]), label=score, marker='o')
            ax.set_xticks(r)
            ax.set_xticklabels(a.index)
            if not numeric:
                ax.tick_params(axis='x', labelrotation=45)
        else:  # Numérique et plus de 15 valeurs
            for score in dico['scoring']:
                a = results.groupby(f'param_{param}').mean()
                a.sort_index(inplace=True, ascending=True)
                ax.plot(a.index, list(a[f"mean_test_{score}"]), label=score)
            if param in xscale:
                ax.set_xscale(xscale[param])

        ax.set_xlabel(param)
        ax.set_ylabel("score")
        ax.legend()

    if len(list_param) % ncols != 0:
        if len(list_param) > ncols:
            for i in range(len(list_param) % ncols, ncols):
                axes[-1, i].set_visible(False)
        elif len(list_param) > 1:
            for i in range(len(list_param) % ncols, ncols):
                axes[i].set_visible(False)

    fig.tight_layout()
    if pdf:
        pdf.export()
    fig.suptitle(f"{dico['model_name']} : effet des paramètres", fontsize=14, y=1)
    fig.tight_layout()
    plt.show()


def key_class_weight(string):
    if string == 'None':
        return -2
    if string == 'balanced':
        return -1
    else:
        dico = eval(string)
        return dico[1] / dico[0]


key_class_weight = np.vectorize(key_class_weight)


def best_score_CV(dico, results, score, display_table=True):
    results_sort = results.sort_values(by=f'mean_test_{score}', ascending=False)

    if display_table:
        display(results_sort.head(10))

    best_params = results_sort.iloc[0].params
    print(f"Meilleure combinaison de paramètres pour {score} :")
    display(best_params)

    return best_params


def graph_2scores_CV_comp(dr_list, score1, score2, s=20, alpha=1, zoom=1, pdf=None):
    """
    Comparaison de plusieurs modèles
    dr_list : [(dico1, results1), (dico2, results2)]
    """
    plt.figure(figsize=(14, 8))

    if isinstance(s, float) or isinstance(s, int):
        s = [s] * len(dr_list)

    for i, elem in enumerate(dr_list):
        dico, results = elem

        if dico['type'] == 'RandomizedSearchCV':
            n = int(zoom * dico['n_iter'])
        else:
            n = int(zoom * dico['len_grid'])

        results_sort = results.sort_values(by=f'mean_test_{score1}', ascending=False)
        plt.scatter(results_sort[f'mean_test_{score1}'][:n], results_sort[f'mean_test_{score2}'][:n], marker='o', s=s[i], alpha=alpha, label=dico['model_name'])

    plt.xlabel(score1)
    plt.ylabel(score2)

    legend = plt.legend()
    for lh in legend.legendHandles:
        lh.set_alpha(1.0)
        lh.set_sizes([30])
    if pdf:
        pdf.export()
    plt.show()


class PDF():
    def __init__(self, fig_folder):
        self.fig_folder = fig_folder
        self.fig_count = 0

        shutil.rmtree(fig_folder, ignore_errors=True)
        os.makedirs(fig_folder)

    def export(self):
        file = f"{self.fig_folder}{self.fig_count:02d}.pdf"
        plt.savefig(file, bbox_inches='tight')
        print(f"Export PDF : {file}\n")        
        self.fig_count += 1
        
        
def convert_keys_to_int(d: dict):
    new_dict = {}
    for k, v in d.items():
        try:
            new_key = int(k)
        except ValueError:
            new_key = k
        if type(v) == dict:
            v = convert_keys_to_int(v)
        new_dict[new_key] = v
    return new_dict