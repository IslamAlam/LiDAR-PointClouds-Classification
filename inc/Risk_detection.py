#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 22:02:10 2018
    Based on: M. Weinmann, B. Jutzi, and C. Mallet (2014)
    Adaptation from: http://www.ipf.kit.edu/code.php#3d_scene
Based on Martin Weinmann
@author: haroldfmurcia, Duvier Lugo, Wiford Mayorga
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from scipy.stats import norm

import Features3D

path_data = os.getcwd()

def plot_neighborhood(X,Y,Z,opt_nn_size):
    # nube de puntos 3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X,Y,Z,s=.01,c=opt_nn_size[:,0])
    plt.xlabel('X')
    plt.ylabel('Y')
    axes = plt.gca()
    plt.show()

def val_Scores(Q,pred_1,pred_2,pred_3,pred_4,pred_5):
    f1_RF   = f1_score(Q, pred_1, average=None)
    f1_SVM  = f1_score(Q, pred_2, average=None)
    f1_NN   = f1_score(Q, pred_3, average=None)
    f1_bdt_real  = f1_score(Q, pred_4, average=None)
    f1_gauss= f1_score(Q, pred_5, average=None)
    f1_post = -np.ones([5,6]) 
    f1_post[0,0:len(f1_RF)] =np.array([f1_RF])
    f1_post[1,0:len(f1_SVM)] =np.array([f1_SVM])
    f1_post[2,0:len(f1_NN)] =np.array([f1_NN])
    f1_post[3,0:len(f1_bdt_real)] =np.array([f1_bdt_real])
    f1_post[4,0:len(f1_gauss)] =np.array([f1_gauss])
    classifiers = ["RF:","SVM:","NN:","bdt_R:","gauss:" ]
    print "-------------------------------------------------------------------"
    print "\t" + ":Ground:" + "\t" + ":RISK 1:" +"\t" + ":RISK 2:" + "\t" + ":WALL:" + "\t"+"\t" + ":others1:" +"\t"+ ":others2:"
    M = f1_post.shape[0]
    for m in range(0,M):
        print classifiers[m] +"\t"+ str(f1_post[m,0]) + "\t" + str(f1_post[m,1]) + "\t" + str(f1_post[m,2]) + "\t" + str(f1_post[m,3]) + "\t" + str(f1_post[m,4])+ str(f1_post[m,5]) 
    print "-------------------------------------------------------------------" + "\n"
    return f1_post

def plot_features(X,Y,Z,features):
    for k in range(0,26):
        fig = plt.figure()
        ax = Axes3D(fig)
        p = ax.scatter(X,Y,Z,s=.01,c=features[:,k],cmap=cm.jet)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(p)
        axes = plt.gca()
        if k==0:
            titulo=('linearity')
        elif k==1:
            titulo=('planarity')
        elif k==2:
            titulo=('scattering')
        elif k==3:
            titulo=('omnivariance')
        elif k==4:
            titulo=('anisotropy')
        elif k==5:
            titulo=('eigenentropy')
        elif k==6:
            titulo=('sum EVs')
        elif k==7:
            titulo=('change of curvature')
        elif k==8:
            titulo=('Z vals')
        elif k==9:
            titulo=('radius kNN')
        elif k==10:
            titulo=('density')
        elif k==11:
            titulo=('verticality')
        elif k==12:
            titulo=('delta Z kNN')
        elif k==13:
            titulo=('std Z kNN')
        elif k==14:
            titulo=('radius kNN 2D')
        elif k==15:
            titulo=('density 2D')
        elif k==16:
            titulo=('sum EVs 2D')
        elif k==17:
            titulo=('EV ratio')
        elif k==18:
            titulo=('frequency acc map')
        elif k==19:
            titulo=('delta z')
        elif k==20:
            titulo=('std z')
        elif k==21:
            titulo=('Evs in 3D 1')
        elif k==22:
            titulo=('Evs in 3D 2')
        elif k==23:
            titulo=('Evs in 3D 3')
        elif k==24:
            titulo=('EVs in 2D 1')
        elif k==25:
            titulo=('EVs in 2D 2')
        plt.show()
        plt.title = titulo
        fileTitle = titulo+'.png'
        fig.savefig(fileTitle)
    
def plot_descriptors(aux_features,aux_Q):
    medias = np.zeros([1,26])
    desviaciones = np.zeros([1,26])
    for k in range(0,26):
        fig = plt.figure()
        if k==0:
            titulo=('linearity')
        elif k==1:
            titulo=('planarity')
        elif k==2:
            titulo=('scattering')
        elif k==3:
            titulo=('omnivariance')
        elif k==4:
            titulo=('anisotropy')
        elif k==5:
            titulo=('eigenentropy')
        elif k==6:
            titulo=('sum EVs')
        elif k==7:
            titulo=('change of curvature')
        elif k==8:
            titulo=('Z vals')
        elif k==9:
            titulo=('radius kNN')
        elif k==10:
            titulo=('density')
        elif k==11:
            titulo=('verticality')
        elif k==12:
            titulo=('delta Z kNN')
        elif k==13:
            titulo=('std Z kNN')
        elif k==14:
            titulo=('radius kNN 2D')
        elif k==15:
            titulo=('density 2D')
        elif k==16:
            titulo=('sum EVs 2D')
        elif k==17:
            titulo=('EV ratio')
        elif k==18:
            titulo=('frequency acc map')
        elif k==19:
            titulo=('delta z')
        elif k==20:
            titulo=('std z')
        elif k==21:
            titulo=('Evs in 3D 1')
        elif k==22:
            titulo=('Evs in 3D 2')
        elif k==23:
            titulo=('Evs in 3D 3')
        elif k==24:
            titulo=('EVs in 2D 1')
        elif k==25:
            titulo=('EVs in 2D 2')
        if k == 11:
            aux_features[:,k]=np.sin(aux_features[:,k]*np.pi/2.0)
        A = aux_features[np.where(aux_Q==0),k]
        B = aux_features[np.where(aux_Q==1),k]
        C = aux_features[np.where(aux_Q==2),k]
        D = aux_features[np.where(aux_Q==3),k]
        E = aux_features[np.where(aux_Q==4),k]
        F = aux_features[np.where(aux_Q==5),k]
        MD0 = np.nanmean(A)
        MD1 = np.nanmean(B)
        MD2 = np.nanmean(C)
        MD3 = np.nanmean(D)
        MD4 = np.nanmean(E)
        MD5 = np.nanmean(F)
        medias[0,k]=np.nanmax(np.array([MD0,MD1,MD2,MD3,MD4,MD5]))
        SD0 = np.std(A)
        SD1 = np.std(B)
        SD2 = np.std(C)
        SD3 = np.std(D)
        SD4 = np.std(E)
        SD5 = np.std(F)
        desviaciones[0,k]=np.nanmax(np.array([SD0,SD1,SD2,SD3,SD4,SD5]))
        N0  = norm(loc = MD0, scale=SD0)
        N1  = norm(loc = MD1, scale=SD1)
        N2  = norm(loc = MD2, scale=SD2)
        N3  = norm(loc = MD3, scale=SD3)
        N4  = norm(loc = MD4, scale=SD4)
        N5  = norm(loc = MD4, scale=SD5)
        x_max = medias[0,k]*2
        if k==8:
            x_max = 2.5
            xmin = -2.5
        else:
            xmin = 0
        x_delta = x_max/100.0
        x1 = np.arange(xmin, x_max, x_delta)
        #plot the pdfs of these normal distributions
        #plt.figure(1, figsize=(8, 6))
        #plt.clf() 
        plt.plot(x1, N0.pdf(x1),label='Clase 0')
        plt.plot(x1, N1.pdf(x1),label='Clase 1')
        plt.plot(x1, N2.pdf(x1),label='Clase 2')
        plt.plot(x1, N3.pdf(x1),label='Clase 3')
        plt.plot(x1, N4.pdf(x1),label='Clase 4')
        plt.plot(x1, N4.pdf(x1),label='Clase 5')
        plt.legend()
        plt.title = titulo
        fileTitle = titulo+'_D.png'
        plt.savefig(fileTitle)
        
def plot_prediction(X,Y,Z,pred,top,title): 
    fig = plt.figure()
    ax = Axes3D(fig)
    P0 = np.where(pred==0)
    P1 = np.where(pred==1)
    P2 = np.where(pred==2)
    P3 = np.where(pred==3)
    P4 = np.where(pred==4)
    P5 = np.where(pred==5)
    if len(P0[0])>1:
        ax.plot(X[P0],Y[P0],Z[P0],'.k',label='Clase 0')
    if len(P1[0])>1:
        ax.plot(X[P1],Y[P1],Z[P1],'.g',label='Clase 1')
    if len(P2[0])>1:
        ax.plot(X[P2],Y[P2],Z[P2],'.r',label='Clase 2')
    if len(P3[0])>1:
        ax.plot(X[P3],Y[P3],Z[P3],'.c',label='Clase 3')
    if len(P4[0])>1:
        ax.plot(X[P4],Y[P4],Z[P4],'.b',label='Clase 4')
    if len(P5[0])>1:
        ax.plot(X[P5],Y[P5],Z[P5],'.b',label='Clase 5')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    axes = plt.gca()
    if top==1:
        ax.view_init(90, 90)
    plt.show()     
    plt.title = title
    fileTitle = title+'_V.png'
    plt.savefig(fileTitle)

if __name__ == "__main__":
    raster_size = 0.5
    kmin = 50
    kmax = 1000
    deltaK = 10
    sub = 100
    selected_features = [2,3,4,7,8,11,12,13,16,17,19,20,23]
    Feat_3D = Features3D.point_class()
    # Features A
    fileName = '/data/a_perro_E.txt' 
    print fileName
    X,Y,Z,R,G,B,Q = Feat_3D.import_data(path_data+fileName)
    X = X[::sub]
    Y = Y[::sub]
    Z = Z[::sub] - np.min(Z)-0.01
    Q1 = Q[::sub]
    pq1=np.where(Q1==1)
    Q1[pq1]=0
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_1 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)
    #plot_descriptors(features_1,Q1)
    #plot_features(X,Y,Z,features_1)
    features_1 = features_1[:,selected_features]
    # Features B
    fileName = '/data/b_MA_park_01_E.txt' 
    print fileName
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName)
    X = X[::sub]
    Y = Y[::sub]
    Z = Z[::sub]- np.min(Z)-0.07
    Q2 = Q[::sub]
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_2 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    #plot_features(X,Y,Z,features_2)
    #plot_descriptors(features_2,Q2)
    # Features D
    fileName = '/data/d_perro_trasero_E.txt' 
    print fileName
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName)
    X = X[::sub]
    Y = Y[::sub]
    Z = Z[::sub]- np.min(Z)
    Q4 = Q[::sub]
    pq4=np.where(Q4==1)
    Q4[pq4]=0
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_4 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    #plot_features(X,Y,Z,features_4)
    #plot_descriptors(features_4,Q4)
    # Features E
    fileName = '/data/e_MA_night_E.txt' 
    print fileName
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName)
    X = X[::sub]
    Y = Y[::sub]
    Z = Z[::sub]- np.min(Z)-0.27
    Q5 = Q[::sub]
    pq5=np.where(Q5==4)
    Q5[pq5]=3
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_5 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    # Features F
    fileName = '/data/f_hueco1_E.txt' 
    print fileName
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName)
    X = X[::sub]
    Y = Y[::sub]
    Z = Z[::sub]- np.min(Z)-0.12
    Q6 = Q[::sub]
    pq6=np.where(Q6==4)
    Q6[pq6]=5
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_6 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    # Features H
    fileName = '/data/h_no_perro_E.txt' 
    print fileName
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName)
    X = X[::sub]
    Y = Y[::sub]
    Z = Z[::sub]- np.min(Z)-0.05
    Q8 = Q[::sub]
    pq8=np.where(Q8==1)
    Q8[pq8]=0
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_8 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    ##########################################################################
    ######################################################   Training data 
    features=np.concatenate((features_1, features_2, features_4, features_5,features_6, features_8), axis=0)
    Q = np.concatenate((Q1, Q2, Q4, Q5, Q6, Q8), axis=0)
    #plot_descriptors(features,Q)
    ##########################################################################
    ######################################################   CLASSIFICATION 
    tic =time.time()
    print "Classification"    
    clf_RF = RandomForestClassifier(max_depth=4, random_state=0)
    clf_RF.fit(features, Q)
    #pred = clf_RF.predict(features)
    #plt.plot(clf.feature_importances_,'.')
    # Clasification SVM
    clf_SVM = LinearSVC(random_state=0)
    clf_SVM.fit(features, Q)
    #pred = clf_SVM.predict(features)
    #plt.plot(clf.feature_importances_,'.')
    clf_NN = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(250, ), random_state=1)
    clf_NN.fit(features, Q)   
    bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=600,
    learning_rate=1)
    bdt_real.fit(features, Q)
    clf_gauss = GaussianNB()
    clf_gauss.fit(features, Q)
    toc =time.time()
    c_time = toc -tic
    ##########################################################################
    ######################################################   VALIDATION
    print "Validation"
    # VALIDATION 1
    fileName_V = '/data/i_MA_park_02_E.txt' 
    print fileName_V
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName_V)
    X = X[::2]
    Y = Y[::2]
    Z = Z[::2]- np.min(Z)-0.02
    Q = Q[::2]
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_V1 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    pred_1 = clf_RF.predict(features_V1)
    pred_2 = clf_SVM.predict(features_V1)
    pred_3 = clf_NN.predict(features_V1)
    pred_4 = bdt_real.predict(features_V1)
    pred_5 = clf_gauss.predict(features_V1)
    V1 = val_Scores(Q,pred_1,pred_2,pred_3,pred_4,pred_5)
    plot_prediction(X,Y,Z,pred_3,0,'V1')
    # VALIDATION 2
    fileName_V = '/data/j_perro_muerto_E.txt' 
    print fileName_V
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName_V)
    X = X[::sub]
    Y = Y[::sub]
    Z = Z[::sub]- np.min(Z)
    Q = Q[::sub]
    p=np.where(Q==1)
    Q[p]=0
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_V2 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    pred_1 = clf_RF.predict(features_V2)
    pred_2 = clf_SVM.predict(features_V2)
    pred_3 = clf_NN.predict(features_V2)
    pred_4 = bdt_real.predict(features_V2)
    pred_5 = clf_gauss.predict(features_V2)
    V2 = val_Scores(Q,pred_1,pred_2,pred_3,pred_4,pred_5)
    plot_prediction(X,Y,Z,pred_3,0,'V2')
    # VALIDATION 3
    fileName_V = '/data/k_MA_garage_E.txt' 
    print fileName_V
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName_V)
    X = X[::sub]
    Y = Y[::sub]
    Z = Z[::sub]- np.min(Z)-0.12
    Q = Q[::sub]
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_V3 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    pred_1 = clf_RF.predict(features_V3)
    pred_2 = clf_SVM.predict(features_V3)
    pred_3 = clf_NN.predict(features_V3)
    pred_4 = bdt_real.predict(features_V3)
    pred_5 = clf_gauss.predict(features_V3)
    V3 = val_Scores(Q,pred_1,pred_2,pred_3,pred_4,pred_5)
    plot_prediction(X,Y,Z,pred_4,0,'V3')
    # VALIDATION 4
    fileName_V = '/data/l_hueco2_E.txt' 
    print fileName_V
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName_V)
    X = X[::sub]
    Y = Y[::sub]
    Z = Z[::sub]- np.min(Z)-0.09
    Q = Q[::sub]
    pq6=np.where(Q==4)
    Q[pq6]=5
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_V4 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    pred_1 = clf_RF.predict(features_V4)
    pred_2 = clf_SVM.predict(features_V4)
    pred_3 = clf_NN.predict(features_V4)
    pred_4 = bdt_real.predict(features_V4)
    pred_5 = clf_gauss.predict(features_V4)
    V4 = val_Scores(Q,pred_1,pred_2,pred_3,pred_4,pred_5)
    plot_prediction(X,Y,Z,pred_1,1,'V4')
    #
    # Features G
    fileName = '/data/g_polocho_E.txt' 
    print fileName
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName)
    X = X[::sub]
    Y = Y[::sub]
    Z = Z[::sub]- np.min(Z)-0.1
    Q = Q[::sub]
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_V5 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    pred_1 = clf_RF.predict(features_V5)
    pred_2 = clf_SVM.predict(features_V5)
    pred_3 = clf_NN.predict(features_V5)
    pred_4 = bdt_real.predict(features_V5)
    pred_5 = clf_gauss.predict(features_V5)
    V5 = val_Scores(Q,pred_1,pred_2,pred_3,pred_4,pred_5)
    plot_prediction(X,Y,Z,pred_1,1,'V5')
    #
    # Features C
    fileName = '/data/c_muro_E.txt' 
    print fileName
    X,Y,Z,R,G,B,Q =Feat_3D.import_data(path_data+fileName)
    X = X[::10]
    Y = Y[::10]
    Z = Z[::10]- np.min(Z)-0.25
    Q = Q[::10]
    opt_nn_size = Feat_3D.optNESS(X,Y,Z,kmin,deltaK,kmax)
    nn_size = opt_nn_size
    features_V6 = Feat_3D.geoFEX(X,Y,Z,nn_size,raster_size)[:,selected_features]
    pred_1 = clf_RF.predict(features_V6)
    pred_2 = clf_SVM.predict(features_V6)
    pred_3 = clf_NN.predict(features_V6)
    pred_4 = bdt_real.predict(features_V6)
    pred_5 = clf_gauss.predict(features_V6)
    V6 = val_Scores(Q,pred_1,pred_2,pred_3,pred_4,pred_5)
    plot_prediction(X,Y,Z,pred_5,0,'V6')
    
        
