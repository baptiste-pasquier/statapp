import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn
import xgboost as xgb
import graphviz


columns_quant = ['contextid',
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

columns_cat = ['display_env',
 'target_env',
 'campaignscenario',
 'campaignvertical',
 'is_interstitial',
 'device_type',
 'hour',
 'weekday']


def datasets(df, verbose=True):    
    X_quant = df[columns_quant]
    X_quant_scaled = StandardScaler().fit_transform(X_quant)
    if verbose:
        print(f"\nNombre de variables pour X_quant : {len(X_quant.columns)}\n")
        display(X_quant.columns)

    X_cat = df[columns_cat]
    X_cat = pd.get_dummies(X_cat, columns=columns_cat, drop_first=True)
    X_cat_scaled = StandardScaler().fit_transform(X_cat)
    if verbose:
        print(f"\nNombre de variables pour X_cat : {len(X_cat.columns)}\n")
        display(X_cat.columns)

    X_quant_cat = df[columns_quant + columns_cat]
    X_quant_cat = pd.get_dummies(X_quant_cat, columns=columns_cat, drop_first=True)
    X_quant_cat_scaled = StandardScaler().fit_transform(X_quant_cat)
    if verbose:
        print(f"\nNombre de variables pour X_quant_cat : {len(X_quant_cat.columns)}")

    y = df['is_display_clicked']
    
    return X_quant, X_quant_scaled, X_cat, X_cat_scaled, X_quant_cat, X_quant_cat_scaled, y



class Modelisation():
    def __init__(self, X, y, model, scaling=False):
        if scaling:
            columns = X.columns
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X = pd.DataFrame(data=X, columns=columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=1234)
        
        X_train = pd.DataFrame(data=X_train, columns=X.columns)
        X_test = pd.DataFrame(data=X_test, columns=X.columns)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        probs = model.predict_proba(X_test)[:, 1]
        
        TP = cm[1][1]
        FN = cm[1][0]
        FP = cm[0][1]
        TN = cm[0][0]
        sc_roc_auc = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # Recall
        Recall = TP/(TP+FN)
        # Precision
        Precision = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # F1_Score
        F1 = (2*Precision*Recall)/(Precision+Recall)

        metrics_score = {'f1': F1, 'recall': Recall,'negative predictive value': NPV,'precision': Precision, 'roc_auc': sc_roc_auc}

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.model = model
        self.probs = probs
        self.metrics_score = metrics_score
        self.recall = Recall
        self.X_columns = X.columns

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def show_conf_matrix(self):
        metrics.plot_confusion_matrix(self.model, self.X_test, self.y_test, cmap='Blues')
        plt.show()     
    
    def show_metrics_score(self):
        for key, value in self.metrics_score.items():
            print(f"{key} : {value:.4f}")
            
    def show_ROC(self):
        fpr, tpr, _ = metrics.roc_curve(self.y_test, self.probs)
        plt.plot(fpr, tpr, label=f"{self.model}")
        plt.plot([0, 1], [0, 1], "r-", label='Modèle aléatoire')
        plt.plot([0, 0, 1], [0, 1, 1], 'b-', label='Modèle parfait')
        plt.legend()
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
        dot_data = sklearn.tree.export_graphviz(self.model, out_file=None, 
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