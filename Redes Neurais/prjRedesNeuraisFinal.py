#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:16:05 2019

@author: renhold
"""
def mapeamento(data, atributo):
    atributoMap=dict()
    contador=0
    for i in data[atributo].unique():
        atributoMap[i]=contador
        contador=contador+1
    data[atributo]=data[atributo].map(atributoMap)
def normalizaDados(dtFrame):
    dtFrame.fillna(-1,inplace=True)
    qtdColunas = len(dtFrame.columns)
    for i in range(qtdColunas):
        mapeamento(dtFrame, i) 
import pandas as pd
dfDiabetes = pd.read_excel('pns-pessoa-Diabetes-P_50_50.xlsx', header=None, skiprows=1)
#dfDiabetes = pd.read_excel('pns-pessoa-Diabetes-P_full.xlsx', header=0)
#from pathlib import Path
#profile = dfDiabetes.profile_report(
#        title="depois", correlation_overrides=["recclass"]
#    )
#profile.to_file(output_file=Path("./depois2_report.html"))
normalizaDados(dfDiabetes)
x = dfDiabetes.iloc[:,0:42]
y = dfDiabetes[43]
from sklearn.model_selection import KFold
txAprendizadoDict = {0: 0.01, 1: 0.05, 2: 0.1, 3: 0.05}
txMomentumDict    = {0: 0.01, 1: 0.05, 2: 0.1, 3: 1.0}
cvScoresList = []
cvScoresDict = {}
varMaxIter   = 1000
from sklearn.neural_network import MLPClassifier
for numeroIteracoes in range(len(txAprendizadoDict)):
    txAprendizado = txAprendizadoDict[numeroIteracoes]
    kf = KFold(n_splits=10)
    clf = MLPClassifier(solver='sgd', 
                        alpha=1e-5, 
                        hidden_layer_sizes=(5,), 
                        random_state=1,
                        learning_rate='adaptive', #constant
                        learning_rate_init=txAprendizadoDict[numeroIteracoes], 
                        momentum=txMomentumDict[numeroIteracoes],
                        max_iter=varMaxIter,
                        early_stopping=False,
                        activation='logistic',
                        validation_fraction=0.3,
                        tol=0.000001)
#    clf = MLPClassifier(solver = 'sgd',
#                        hidden_layer_sizes=(5,),
#                        random_state=1,
#                        learning_rate='constant',
#                        learning_rate_init=txAprendizadoDict[numeroIteracoes],
#                        max_iter=varMaxIter,
#                        activation='logistic',
#                        momentum=txMomentumDict[numeroIteracoes],
#                        early_stopping=False,
#                        validation_fraction=0.3,
#                        tol=0.000001
#                        )
    for k, (train, test) in enumerate(kf.split(x,y)):
        clf.fit(x.iloc[train], y.iloc[train])
        cvScoresDict = {}
        cvScoresDict['fold'] = k
        cvScoresDict['taxa acerto'] = clf.score(x.iloc[test], y.iloc[test])
        cvScoresDict['taxa aprendizado'] = txAprendizadoDict[numeroIteracoes]
        cvScoresDict['taxa momentum'] = txMomentumDict[numeroIteracoes]
        cvScoresDict['max_iter'] = varMaxIter 
        cvScoresList.append(cvScoresDict)
max_scores = {'score1':0, 'scoreDesc1': '',
              'score2':0, 'scoreDesc2': '',
              'score3':0, 'scoreDesc3': '',
              'scoreMax':0, 'scoreDescMax': ''}
for i in range(len(cvScoresList)):
    msg = "fold: {0}, tx acerto: {1: .5f}, tx aprendizado: {2}, tx momentum: {3}, max_iter: {4}"
    msgFormatada = msg.format(cvScoresList[i]['fold'], 
                              cvScoresList[i]['taxa acerto'], 
                              cvScoresList[i]['taxa aprendizado'], 
                              cvScoresList[i]['taxa momentum'],
                              cvScoresDict['max_iter'])
    if cvScoresList[i]['taxa acerto'] > max_scores['scoreMax']: 
        max_scores['scoreMax'] = cvScoresList[i]['taxa acerto']
        max_scores['scoreDescMax'] = msgFormatada         
    if i <= 9:
        if cvScoresList[i]['taxa acerto'] > max_scores['score1']: 
            max_scores['score1'] = cvScoresList[i]['taxa acerto']
            max_scores['scoreDesc1'] = msgFormatada         
    elif i <= 19:
        if cvScoresList[i]['taxa acerto'] > max_scores['score2']: 
            max_scores['score2'] = cvScoresList[i]['taxa acerto']
            max_scores['scoreDesc2'] = msgFormatada 
    else:    
        if cvScoresList[i]['taxa acerto'] > max_scores['score3']: 
            max_scores['score3'] = cvScoresList[i]['taxa acerto']
            max_scores['scoreDesc3'] = msgFormatada         
print('Melhor teste 1: {0}'.format(max_scores['scoreDesc1']))
print('Melhor teste 2: {0}'.format(max_scores['scoreDesc2']))
print('Melhor teste 3: {0}'.format(max_scores['scoreDesc3']))        
print('Melhor teste: {0}'.format(max_scores['scoreDescMax']))
