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

from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, Matern

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.model_selection import StratifiedKFold

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
    
    best_columns = np.array(list(df.columns))[best_indices]
    best_features = np.array(list(df[best_columns].values))
    
    sel_ = SelectFromModel(LogisticRegression(C=0.8, penalty='l1', solver="liblinear"))
    sel_.fit(best_features, y)
    selected_features_bool = sel_.get_support()
    
    final_selected=[]
    for index,feat_id in enumerate(best_columns):
        if selected_features_bool[index]:
            final_selected.append(feat_id)    
    print("Best l1 features: "+str(len(final_selected)))
    
    return np.array(final_selected)
    
def machine_learning(dataset_, labels_, split, classifier,radiogenomics=False):
    if not radiogenomics:    
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
        
        oversample = SMOTE(sampling_strategy=1)
        train_set, train_labels = oversample.fit_resample(train_set, train_labels)
    else: # radiogenomics
        r_train_pids = split["radiomics"][0]
        r_test_pids = split["radiomics"][1]
        g_train_pids = split["genomics"][0]
        g_test_pids = split["genomics"][1]
        
        r_train_set = []
        g_train_set = []
        train_labels=[]
        for key in r_train_pids:
            try:
                train_labels.append(labels_[key])
                r_train_set.append(dataset_["radiomics"][key])
                g_train_set.append(dataset_["genomics"][key])
            except:
                # print(key+" not available features IN TRAINING SET!")   
                continue             
        r_train_set = np.array(r_train_set)
        g_train_set = np.array(g_train_set)
        train_labels = np.stack(train_labels)
        
        oversample = SMOTE(sampling_strategy=1)
        r_train_set, f_train_labels = oversample.fit_resample(r_train_set, train_labels)
        g_train_set, f_train_labels = oversample.fit_resample(g_train_set, train_labels)
        
        test_pids=[]
        for key in r_train_pids:
            if key in g_train_pids:
                test_pids.append(key)
                
        train_set = np.concatenate((r_train_set, g_train_set),axis=1)
        train_labels = f_train_labels
        test_pids = list(set(r_test_pids) | set(g_test_pids)) 
    test_set = []
    test_labels =[]
    for key in test_pids:
        try:
            test_labels.append(labels_[key])
            if radiogenomics:
                r_pattern = dataset_["radiomics"][key]
                g_pattern = dataset_["genomics"][key]
                test_set.append(np.concatenate((r_pattern,g_pattern),axis=0))
            else:
                test_set.append(dataset_[key])
        except:
            # print(key+" not available features IN TESTING SET!")
            continue
    test_set = np.array(test_set)
    test_labels = np.stack(test_labels)
    print("Training set: "+str(len(train_labels)))
    print("Testing set: "+str(len(test_labels)))
        
    
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
    # y_pred = np.argmax(y_pred,axis=0)
    
    
    y_score = clf.predict(test_set)
    
    if "svm" in classifier:
        pred = clf.decision_function(test_set)
        score_roc = roc_auc_score(test_labels, pred)
        fpr, tpr, thresholds = roc_curve(test_labels, pred)
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

# pids_volume_information = pd.read_excel(hypes["dataset_dir"]+"segmentation information.xls")

if hypes_nn["image_normalization"] == "-11":
    ss = StandardScaler()
elif hypes_nn["image_normalization"] == "01":
    ss = Normalizer()

#load genomic data
genomics_ = pkl.load(open(hypes["dataset_dir"]+"genomics-RNAseq.pkl","rb"))

genomics = ss.fit_transform(genomics_)
genomics = pd.DataFrame(data=genomics,index=genomics_.index, columns=genomics_.columns)

# #load radiomics data
radiomics_vector = pkl.load(open(hypes["dataset_dir"]+"3d_radiomics_vector_raw_images.pkl", "rb"))
radiomics_ = pd.DataFrame.from_dict(radiomics_vector,orient="index")

radiomics = ss.fit_transform(radiomics_)
radiomics = pd.DataFrame(data=radiomics,index=radiomics_.index, columns=radiomics_.columns)

#training objectives labels
labels  = pkl.load(open(hypes["dataset_dir"]+"end-points.pkl","rb"))
    
splits={}
for experiment in ["SUBTYPES","EGFR", "KRAS"]:
    pids = np.array(list(labels[experiment].keys()),dtype=str)
    f_labels = np.array(list(labels[experiment].values()))
    sss = StratifiedKFold(n_splits=5, shuffle=True)          
    kfolds=[]          
    for train_index, test_index in sss.split(pids, f_labels):
        kfolds.append([pids[train_index], pids[test_index]])
    splits[experiment] = kfolds
pkl.dump(splits,open("chkp/results/splits.pkl","wb"))    

exp_perf={}
results={}
for classifier in ["GPC_RBF", "decision_tree", "KNN","poly_svm", "linear_svm", "rbf_svm", "sigmoid_svm"]:#"GPC_EXP","GPC_MAT", "GPC_DOT", "GPC_EXP","GPC_MAT"]:    
    for experiment in ["SUBTYPES","EGFR", "KRAS"]:
        rg_ktpr=[]    
        g_ktpr=[]    
        r_ktpr=[]    
        rg_kacc=[]
        rg_kauc=[]
        rg_ksn=[]
        rg_ksp=[]
        g_kacc=[]
        g_kauc=[]
        g_ksn=[]
        g_ksp=[]
        r_kacc=[]
        r_kauc=[]
        r_ksn=[]
        r_ksp=[]
        for index,split in enumerate(splits[experiment]):
            g_tr_split=[]
            g_tst_split=[]
            for key in list(genomics.index):
                if key in list(split[0]):
                    g_tr_split.append(key)
                elif key in list(split[1]):
                    g_tst_split.append(key)
                    
            r_tr_split=[]
            r_tst_split=[]
            for key in list(radiomics.index):
                if key in list(split[0]):
                    r_tr_split.append(key)
                elif key in list(split[1]):
                    r_tst_split.append(key)
                    
            g_labels=[]
            for pid in list(genomics.loc[g_tr_split].index):
                try:
                    g_labels.append(labels[experiment][pid])
                except:
                    continue
            
            r_labels=[]
            for pid in list(radiomics.loc[r_tr_split].index):
                try:
                    r_labels.append(labels[experiment][pid])
                except:
                    continue
                
            print("feature selection")
            
            print("Genomics")
            genomic_feat = apply_feature_selection(genomics.loc[g_tr_split], g_labels, cutoff_pvalue=0.05)
            print("Radiomics")
            radiomic_feat = apply_feature_selection(radiomics.loc[r_tr_split], r_labels, cutoff_pvalue=0.05)
           
            path = "chkp/results/"+experiment+"_"+classifier+"_nsp"+str(index+1)
            os.mkdir(path)
            np.save(path+"/selected_gen",np.array(genomic_feat,dtype=str))
            np.save(path+"/selected_rad",np.array(radiomic_feat,dtype=str))
            
            selected_genomic={}
            for key in list(genomics.index):
                selected_genomic[key] = genomics[genomic_feat].loc[key].to_numpy()
                
            selected_radiomics={}
            for key in list(radiomics.index):
                selected_radiomics[key] = radiomics[radiomic_feat].loc[key].to_numpy()
                
        #Radiomics
            combined_patterns={}
            for key in list(selected_radiomics.keys()):
                try:
                    combined_patterns[key] = np.concatenate((selected_genomic[key],selected_radiomics[key]))
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
            for key in sorted(selected_radiomics.keys()):
                try:
                    r_labels[key]=labels[experiment][key]
                    r_patterns[key]=selected_radiomics[key]
                except:
                    print(key+" not labeled!")
                    continue

            print("classification: "+classifier)
            rg_acc, rg_auc, rg_sn, rg_sp, rg_tpr = machine_learning({"radiomics":r_patterns,"genomics":g_patterns}, rg_labels, {"radiomics":[r_tr_split,r_tst_split],"genomics":[g_tr_split,g_tst_split]}, classifier, radiogenomics=True)
            r_acc, r_auc, r_sn, r_sp, r_tpr = machine_learning(r_patterns, rg_labels, [r_tr_split,r_tst_split], classifier, radiogenomics=False)
            g_acc, g_auc, g_sn, g_sp, g_tpr = machine_learning(g_patterns, rg_labels, [g_tr_split,g_tst_split], classifier, radiogenomics=False)
            
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
        
        g_kacc=np.array(g_kacc)
        g_kauc=np.array(g_kauc)
        g_ksn=np.array(g_ksn)
        g_ksp=np.array(g_ksp)
        
        r_kacc=np.array(r_kacc)
        r_kauc=np.array(r_kauc)
        r_ksn=np.array(r_ksn)
        r_ksp=np.array(r_ksp)
        
        plot_roc(rg_ktpr,np.array(rg_kauc).mean(),title="radiogenomics "+classifier+" "+experiment)
        plot_roc(g_ktpr,np.array(g_kauc).mean(),title="genomics "+classifier+" "+experiment)
        plot_roc(r_ktpr,np.array(r_kauc).mean(),title="radiomics "+classifier+" "+experiment)

        results["radiogenomics "+classifier+" "+experiment] = pd.Series({"ACC_MEAN":rg_kacc.mean(),"ACC_STD":rg_kacc.std(),"AUC_MEAN":rg_kauc.mean(),"AUC_STD":rg_kauc.std(),"SN_MEAN":rg_ksn.mean(),"SN_STD":rg_ksn.std(),"SP_MEAN":rg_ksp.mean(),"SP_STD":rg_ksp.std()})
        results["radiomics "+classifier+" "+experiment] = pd.Series({"ACC_MEAN":r_kacc.mean(),"ACC_STD":r_kacc.std(),"AUC_MEAN":r_kauc.mean(),"AUC_STD":r_kauc.std(),"SN_MEAN":r_ksn.mean(),"SN_STD":r_ksn.std(),"SP_MEAN":r_ksp.mean(),"SP_STD":r_ksp.std()})
        results["genomics "+classifier+" "+experiment] = pd.Series({"ACC_MEAN":g_kacc.mean(),"ACC_STD":g_kacc.std(),"AUC_MEAN":g_kauc.mean(),"AUC_STD":g_kauc.std(),"SN_MEAN":g_ksn.mean(),"SN_STD":g_ksn.std(),"SP_MEAN":g_ksp.mean(),"SP_STD":g_ksp.std()})
        
        
final_results = pd.DataFrame.from_dict(results, orient="index")
final_results.to_csv("chkp/results/final_results.csv")