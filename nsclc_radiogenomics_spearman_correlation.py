#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: trivizakis

@github: github.com/trivizakis
"""
import numpy as np
import pickle as pkl

from sklearn.preprocessing import StandardScaler

from scipy.stats import spearmanr

from matplotlib import pyplot as plt

def significant_rho(correlation, feature_names, p_value=0.05, rho_thresshold=0.8):    
    pvalues = correlation[1]
    pvalues[pvalues<p_value]=1
    pvalues[pvalues<1]=0
    
    rho = correlation[0]*pvalues
    rho[rho<rho_thresshold]=0
    
    return rho

#load features from pickle (dictionaries)
with open("dataset/3d_radiomics_vector_raw_images.pkl", "rb") as file:
    radiomics_vector = pkl.load(file)
    
with open("dataset/genomics-RNAseq.pkl", "rb") as file:
    genomics_vector = pkl.load(file)


for pid in list(radiomics_vector.keys()):
    try:
        genomics_vector.loc[pid]
    except:
        del(radiomics_vector[pid])
        
for pid in list(genomics_vector.index):
    try:
        radiomics_vector[pid]
    except:
       genomics_vector=genomics_vector.drop(pid)
        
ss = StandardScaler()
frd = ss.fit_transform(np.array(list(radiomics_vector.values())))    
fgd = ss.fit_transform(genomics_vector.values)   
    
   
# pc_radiomics = spearmanr(frd,frd, axis=0)
# pc_genomics = spearmanr(fgd,fgd, axis=0)

pc_radiogenomics = spearmanr(frd,fgd, axis=0)

# rho_radiomics = significant_rho(pc_radiomics, feature_names=list(radiomics_vector["R01-029"].index), p_value=0.05, rho_thresshold=0.5)
# rho_genomics = significant_rho(pc_genomics, feature_names=list(genomics_vector.columns), p_value=0.05, rho_thresshold=0.5)

rho_radiogenomics = significant_rho(pc_radiogenomics, feature_names=list(radiomics_vector["R01-029"].index)+list(genomics_vector.columns), p_value=0.05, rho_thresshold=0.6)

figure = plt.figure()
axes = figure.add_subplot(111)
caxes = axes.matshow(rho_radiogenomics, cmap="gist_heat")
figure.colorbar(caxes)
figure.savefig("spearman radiogenomics.png",dpi=300)
