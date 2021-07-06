#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: trivizakis

@github: github.com/trivizakis
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, Matern

from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from matplotlib import pyplot as plt

from sklearn.feature_selection import f_classif as fc
from sklearn.feature_selection import SelectKBest as kbest

from numpy import interp
import scipy.stats as st

sys.path.append('../cnn_framework')
from utils import Utils

def plot_roc(tprs, mean_auc, title):
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    
    base_fpr = np.linspace(0, 1, 101)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    
    plt.plot(base_fpr, mean_tprs, 'b', label="Mean AUC=%.2f" % mean_auc)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(title)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("chkp/figs/"+title+".tiff", dpi=300)
    plt.show()
        
def apply_feature_selection(df, labels, cutoff_pvalue=0.05):
    X=[]
    for key in list(df.index):
        X.append(df.loc[key])
    X = np.array(X)
    y = np.hstack(labels)
    
    selector = kbest(fc, k="all")
    best_features = selector.fit_transform(X, y)
    f_scores, p_values = fc(X, y)
    critical_value = st.f.ppf(q=1-cutoff_pvalue, dfn=len(np.unique(y))-1, dfd=len(y)-len(np.unique(y)))
    
    best_indices=[]
    for index, p_value in enumerate(p_values):
        if f_scores[index]>critical_value and p_value<cutoff_pvalue:
            best_indices.append(index)
    print("Best ANOVA features: "+str(len(best_indices)))
    
    if len(best_features)>0:
        best_columns = np.array(list(df.columns))[best_indices]
        best_features = np.array(list(df[best_columns].values))
    else:
        best_columns = np.array(list(df.columns))
        best_features = np.array(list(df.values))
    
    try:
        sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver="liblinear"))
        sel_.fit(best_features, y)
        selected_features_bool = sel_.get_support()
        
        final_selected=[]
        for index,feat_id in enumerate(best_columns):
            if selected_features_bool[index]:
                final_selected.append(feat_id)    
        final_selected = np.array(final_selected)
    except:
        print("No features left after L1")
        final_selected = best_columns
        
    print("Best l1 features: "+str(len(final_selected)))
    
    return final_selected

def machine_learning(dataset_, labels_, split, classifier):    
    train_pids = split[0]
    test_pids = split[1]
    
    train_set = []
    train_labels =[]
    for key in train_pids:
        try:
            train_set.append(dataset_[key])
            train_labels.append(labels_[key])
        except:
            print(key+" not available features IN TRAINING SET!")   
            continue             
    train_set = np.array(train_set)
    train_labels = np.stack(train_labels)
    
    test_set = []
    test_labels =[]
    for key in test_pids:
        try:
            test_set.append(dataset_[key])
            test_labels.append(labels_[key])
        except:
            print(key+" not available features IN TESTING SET!")
            continue
    test_set = np.array(test_set)
    test_labels = np.stack(test_labels)
    
    if classifier == "poly_svm":
        clf = svm.SVC(kernel="poly", gamma="auto", probability=True)
    elif classifier == "linear_svm":
        clf = svm.SVC(kernel="linear", gamma="auto", probability=True)
    elif classifier == "rbf_svm":
        clf = svm.SVC(kernel="rbf", gamma="auto", probability=True)
    elif classifier == "sigmoid_svm":
        clf = svm.SVC(kernel="sigmoid", gamma="auto", probability=True)
    elif classifier == "decision_tree":
        clf = tree.DecisionTreeClassifier()
    elif classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=10)
    elif classifier == "GPC_RBF":
        kernel = 1.0 * RBF(length_scale=1.0)
        clf = GaussianProcessClassifier(kernel=kernel)
    elif classifier == "GPC_DOT":
        kernel = 1.0 * DotProduct(sigma_0=1.0)**2
        clf = GaussianProcessClassifier(kernel=kernel)
    elif classifier == "GPC_EXP":
        kernel = ExpSineSquared()
        clf = GaussianProcessClassifier(kernel=kernel)
    elif classifier == "GPC_MAT":
        kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        clf = GaussianProcessClassifier(kernel=kernel)
        
    clf = clf.fit(train_set,train_labels)
    acc = clf.score(test_set,test_labels)
    y_pred = clf.predict_proba(test_set)
    
    y_score = clf.predict(test_set)
    
    if "svm" in classifier:
        pred = clf.decision_function(test_set)
        fpr, tpr, thresholds = roc_curve(test_labels, pred)
        score_roc = roc_auc_score(test_labels, pred)
    else:
        score_roc = roc_auc_score(test_labels, y_pred[:,1])
        fpr, tpr, _ = roc_curve(test_labels,y_pred[:,1])
    
    tn, fp, fn, tp = confusion_matrix(test_labels, y_score).ravel()
    
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    
    base_fpr = np.linspace(0, 1, 101)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    return acc, score_roc, sn, sp, tpr

#genomic
hypes = Utils.get_hypes("hypes")
hypes_nn = Utils.get_hypes("hypes_nn")

if hypes_nn["image_normalization"] == "-11":
    ss = StandardScaler()
elif hypes_nn["image_normalization"] == "01":
    ss = Normalizer()

#load genomic data
genomics_ = pkl.load(open(hypes["dataset_dir"]+"genomics-RNAseq.pkl","rb"))

genomics = ss.fit_transform(genomics_)
genomics = pd.DataFrame(data=genomics,index=genomics_.index, columns=genomics_.columns)

#load deep_feratures data
deep = pkl.load(open("chkp/deep features/deep_features_raw.pkl","rb"))

#training objectives labels
labels  = pkl.load(open(hypes["dataset_dir"]+"end-points.pkl","rb"))

splits={}
for experiment in ["SUBTYPES","EGFR", "KRAS"]:
    pids = np.array(list(labels[experiment].keys()),dtype=str)
    # set_ = np.array(list(dataset_.values()))
    f_labels = np.array(list(labels[experiment].values()))
    sss = StratifiedKFold(n_splits=5, shuffle=True)          
    kfolds=[]          
    for train_index, test_index in sss.split(pids, f_labels):
        kfolds.append([pids[train_index], pids[test_index]])
    splits[experiment] = kfolds
pkl.dump(splits,open("chkp/results/splits.pkl","wb"))    

exp_perf={}
results={}
failed=[]
for model_name in list(deep.keys()):
    for classifier in ["GPC_RBF", "decision_tree", "KNN","poly_svm", "linear_svm", "rbf_svm", "sigmoid_svm"]:#"GPC_EXP","GPC_MAT", "GPC_DOT", "GPC_EXP","GPC_MAT"]:    
        for experiment in ["SUBTYPES","EGFR", "KRAS"]:
            rg_ktpr=[]
            rg_kacc=[]
            rg_kauc=[]
            rg_ksn=[]
            rg_ksp=[]
            g_ktpr=[]
            g_kacc=[]
            g_kauc=[]
            g_ksn=[]
            g_ksp=[]
            r_ktpr=[]
            r_kacc=[]
            r_kauc=[]
            r_ksn=[]
            r_ksp=[]
            for index,split in enumerate(splits[experiment]):
                
                print("feature selection, deep: "+model_name+" clf: "+classifier+" exp: "+experiment)
                g_tr_split=[]
                g_tst_split=[]
                for key in list(genomics.index):
                    if key in list(split[0]):
                        g_tr_split.append(key)
                    elif key in list(split[1]):
                        g_tst_split.append(key)
                        
                r_tr_split=[]
                r_tst_split=[]
                for key in list(deep[model_name].index):
                    if key in list(split[0]):
                        r_tr_split.append(key)
                    elif key in list(split[1]):
                        r_tst_split.append(key)
                                    
                deep_ = ss.fit_transform(deep[model_name])
                columns_=["f"+str(s) for s in deep[model_name].columns]
                deep_ = pd.DataFrame(data=deep_,index=deep[model_name].index, columns=columns_)#, columns=deep[model_name].columns)
                
                g_labels=[]
                for pid in list(genomics.loc[g_tr_split].index):
                    try:
                        g_labels.append(labels[experiment][pid])
                    except:
                        continue
            
                r_labels=[]
                for pid in list(deep_.loc[r_tr_split].index):
                    try:
                        r_labels.append(labels[experiment][pid])
                    except:
                        continue
                    
                print("feature selection")
                
                try:
                    genomic_feat = apply_feature_selection(genomics.loc[g_tr_split], g_labels)
                    deep_feat = apply_feature_selection(deep_.loc[r_tr_split], r_labels)
                except:
                    failed = experiment+"_"+model_name+"_"+"_"+classifier
                    break
               
                path = "chkp/results/"+experiment+"_"+model_name+"_"+"_"+classifier+"_nsp"+str(index+1)#deep
                os.mkdir(path)
                np.save(path+"/selected_gen",np.array(genomic_feat,dtype=str))
                np.save(path+"/selected_deep",np.array(deep_feat,dtype=str))
                
                selected_genomic={}
                for key in list(genomics.index):
                    selected_genomic[key] = genomics[genomic_feat].loc[key].to_numpy()
                 
                selected_deep={}
                for key in list(deep_.index):
                    selected_deep[key] = deep_[deep_feat].loc[key].to_numpy()

                combined_patterns={}
                for key in list(selected_deep.keys()):
                    try:
                        combined_patterns[key] = np.concatenate((selected_genomic[key],selected_deep[key]))#deep
                    except:
                        print(key)
                        continue
                    
                rg_labels={}
                rg_patterns={}
                for key in sorted(combined_patterns.keys()):
                    try:
                        rg_labels[key]=labels[experiment][key]
                        rg_patterns[key]=combined_patterns[key]
                    except:
                        print(key+" not labeled!")
                        continue
                    
                g_labels={}
                g_patterns={}
                for key in sorted(selected_genomic.keys()):
                    try:
                        g_labels[key]=labels[experiment][key]
                        g_patterns[key]=selected_genomic[key]
                    except:
                        print(key+" not labeled!")
                        continue
                        
                r_labels={}
                r_patterns={}
                for key in sorted(selected_deep.keys()):
                    try:
                        r_labels[key]=labels[experiment][key]
                        r_patterns[key]=selected_deep[key]
                    except:
                        print(key+" not labeled!")
                        continue
                        
                print("classification: "+classifier)
                
                try:
                    rg_acc, rg_auc, rg_sn, rg_sp, rg_tpr = machine_learning(rg_patterns, rg_labels, split, classifier)
                    r_acc, r_auc, r_sn, r_sp, r_tpr = machine_learning(r_patterns, r_labels, [r_tr_split,r_tst_split], classifier)
                    g_acc, g_auc, g_sn, g_sp, g_tpr = machine_learning(g_patterns, g_labels, [g_tr_split,g_tst_split], classifier)
                except:
                    print("Something went wrong!")
                    continue
            
                rg_kacc.append(rg_acc)
                rg_kauc.append(rg_auc)
                rg_ksn.append(rg_sn)
                rg_ksp.append(rg_sp)
                rg_ktpr.append(rg_tpr)                
                
                g_kacc.append(g_acc)
                g_kauc.append(g_auc)
                g_ksn.append(g_sn)
                g_ksp.append(g_sp)
                g_ktpr.append(g_tpr)
                
                r_kacc.append(r_acc)
                r_kauc.append(r_auc)
                r_ksn.append(r_sn)
                r_ksp.append(r_sp)
                r_ktpr.append(r_tpr)
                
            rg_kacc=np.array(rg_kacc)
            rg_kauc=np.array(rg_kauc)
            rg_ksn=np.array(rg_ksn)
            rg_ksp=np.array(rg_ksp)
            rg_ktpr=np.array(rg_ktpr)
            
            g_kacc=np.array(g_kacc)
            g_kauc=np.array(g_kauc)
            g_ksn=np.array(g_ksn)
            g_ksp=np.array(g_ksp)
            g_ktpr=np.array(g_ktpr)
            
            r_kacc=np.array(r_kacc)
            r_kauc=np.array(r_kauc)
            r_ksn=np.array(r_ksn)
            r_ksp=np.array(r_ksp)
            r_ktpr=np.array(r_ktpr)
             
            plot_roc(rg_ktpr,np.array(rg_kauc).mean(),title="radiogenomics "+classifier+" "+experiment+" "+model_name)
            plot_roc(g_ktpr,np.array(g_kauc).mean(),title="genomics "+classifier+" "+experiment+" "+model_name)
            plot_roc(r_ktpr,np.array(r_kauc).mean(),title="radiomics "+classifier+" "+experiment+" "+model_name)

            results["deepgenomics "+classifier+" "+experiment+" "+model_name] = pd.Series({"ACC_MEAN":rg_kacc.mean(),"ACC_STD":rg_kacc.std(),"AUC_MEAN":rg_kauc.mean(),"AUC_STD":rg_kauc.std(),"SN_MEAN":rg_ksn.mean(),"SN_STD":rg_ksn.std(),"SP_MEAN":rg_ksp.mean(),"SP_STD":rg_ksp.std()})
            results["deep "+classifier+" "+experiment+" "+model_name] = pd.Series({"ACC_MEAN":r_kacc.mean(),"ACC_STD":r_kacc.std(),"AUC_MEAN":r_kauc.mean(),"AUC_STD":r_kauc.std(),"SN_MEAN":r_ksn.mean(),"SN_STD":r_ksn.std(),"SP_MEAN":r_ksp.mean(),"SP_STD":r_ksp.std()})
            results["genomics "+classifier+" "+experiment+" "+model_name] = pd.Series({"ACC_MEAN":g_kacc.mean(),"ACC_STD":g_kacc.std(),"AUC_MEAN":g_kauc.mean(),"AUC_STD":g_kauc.std(),"SN_MEAN":g_ksn.mean(),"SN_STD":g_ksn.std(),"SP_MEAN":g_ksp.mean(),"SP_STD":g_ksp.std()})
           
        
final_results = pd.DataFrame.from_dict(results, orient="index")
final_results.to_csv("chkp/results/final_results.csv")