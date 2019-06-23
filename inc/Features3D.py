#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 22:02:10 2018
    Based on: M. Weinmann, B. Jutzi, and C. Mallet (2014)
    Adaptation from: http://www.ipf.kit.edu/code.php#3d_scene
@author: haroldfmurcia
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA

path_data = os.getcwd()

class point_class(object):
    def __init__ (self):
        print("Inicialization")
        
    def import_data(self, fileName):
        data = pd.read_csv(fileName,sep="\t", header = None)
        data.columns=["x","y","z","r","g","b","Q"]
        X= np.array(data.x)
        Y= np.array(data.y)
        Z= np.array(data.z)
        R= np.array(data.r)
        G= np.array(data.g)
        B= np.array(data.b)
        Q= np.array(data.Q)
        return X,Y,Z,R,G,B,Q
    
    def optNESS(self, X,Y,Z,kmin,deltaK,kmax):
        XYZ = np.array([X,Y,Z]).T
        k_plus_1 = kmax+1;
        K = np.linspace(kmin, kmax, deltaK)
        # get local neighborhoods consisting of k neighbors
        nbrs = NearestNeighbors(n_neighbors=k_plus_1, algorithm='kd_tree', metric='euclidean').fit(XYZ)
        distances, idx = nbrs.kneighbors(XYZ)
        point_ID_max = len(X)
        num_k = len(K)
        # do some initialization stuff for incredible speed improvement
        Shannon_entropy = np.zeros([point_ID_max,num_k])
        opt_nn_size = np.zeros([point_ID_max,1])
        # calculate Shannon entropy
        for j1 in range(0,point_ID_max):
            Shannon_entropy_real = np.zeros([1,num_k])
            for j2 in range(0,num_k):
                # select neighboring points
                P = idx[j1,1:int(K[j2])+1]          # the point and its k neighbors ...
                cov_mat = np.cov([X[P],Y[P],Z[P]])
                eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
                eig_val_cov = np.sort(eig_val_cov)
                epsilon_to_add = 1e-8;
                if eig_val_cov[2] <=0:
                    eig_val_cov[2] = epsilon_to_add
                    if eig_val_cov[1] <=0:
                        eig_val_cov[1] = epsilon_to_add
                        if eig_val_cov[0] <=0:
                            eig_val_cov[0] = epsilon_to_add
                # normalize EVs
                EVs = 1.0*eig_val_cov/sum(eig_val_cov);
                # derive Shannon entropy based on eigenentropy
                Shannon_entropy_cal = -( EVs[0]*np.log(EVs[0]) + EVs[1]*np.log(EVs[1]) + EVs[2]*np.log(EVs[2]) )
                Shannon_entropy_real[0,j2] = np.real(Shannon_entropy_cal)
            Shannon_entropy[j1,:] = Shannon_entropy_real
            #select k with minimal Shannon entropy
            min_entry_of_Shannon_entropy = np.argmin(Shannon_entropy_real)
            opt_nn_size[j1,0] = K[min_entry_of_Shannon_entropy]
        return opt_nn_size
    
    def geoFEX(self, X,Y,Z,nn_size,raster_size):
        #    INPUT VARIABLES
        #    XYZI          -   matrix containing XYZI [n x 4]
        #    nn_size       -   vector containing the number of neighbors for each 3D point 
        #    raster_size   -   raster size in [m], e.g. 0.25m
        
        # get point IDs
        point_ID_max = X.shape[0]
        # get local neighborhoods consisting of k neighbors (the maximum k value is chosen here in order to conduct knnsearch only once)
        X_vals=np.array([X]).T
        Y_vals=np.array([Y]).T
        Z_vals=np.array([Z]).T
        data_pts = np.array([X_vals[:,0],Y_vals[:,0],Z_vals[:,0]]).T   # XYZ data
        k_plus_1 = int(max(nn_size))+1;
        nbrs = NearestNeighbors(n_neighbors=k_plus_1, algorithm='kd_tree', metric='euclidean').fit(data_pts)
        dist, idx = nbrs.kneighbors(data_pts)
        
        # do some initialization stuff for incredible speed improvement
        normal_vector = np.zeros([point_ID_max,3])
        EVs           = np.zeros([point_ID_max,3])
        sum_EVs       = np.zeros([point_ID_max,1])
        radius_kNN    = np.zeros([point_ID_max,1])
        density       = np.zeros([point_ID_max,1])
        delta_Z_kNN   = np.zeros([point_ID_max,1])
        std_Z_kNN     = np.zeros([point_ID_max,1])
        radius_kNN_2D = np.zeros([point_ID_max,1])
        density_2D    = np.zeros([point_ID_max,1])
        sum_EVs_2D    = np.zeros([point_ID_max,1])
        EVs_2D        = np.zeros([point_ID_max,2])
        
        # loop over all 3D points
        for j1 in range(0,point_ID_max):
            # select neighboring points
            P = idx[j1,0:int(nn_size[j1])+1] 
            #m = XYZ[P].shape[0]
            # calculate covariance matrix C
            cov_mat = np.cov([X_vals[P,0],Y_vals[P,0],Z_vals[P,0]])
            eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)  
            # sorting
            idx_eig = eig_val_cov.argsort()[::1] 
            eig_val_cov = eig_val_cov[idx_eig]
            eig_vec_cov = eig_vec_cov[:,idx_eig]
            epsilon_to_add = 1e-8;
            s1 = eig_val_cov[0]  #Min
            s2 = eig_val_cov[1]
            s3 = eig_val_cov[2]  #Max
            EVs[j1,:]= np.array([s3,s2,s1])
            if EVs[j1,2] <=0:
                EVs[j1,2] = epsilon_to_add
                if EVs[j1,1] <=0:
                    EVs[j1,1] = epsilon_to_add
                    if EVs[j1,0] <=0:
                        EVs[j1,0] = epsilon_to_add
            sum_EVs[j1,0] = np.sum(EVs[j1,:])
    
            # ... as well as normal vectors derived from eigenvectors (columns in V)
            if (s1 <= s2 and s1<= s3):
                normal_vector[j1,:] = eig_vec_cov[:,0]/LA.norm(eig_vec_cov[:,0])
            elif (s2 <= s1 and s2 <= s3):
                normal_vector[j1,:] = eig_vec_cov[:,1]/LA.norm(eig_vec_cov[:,1])
            elif (s3 <= s1 and s3 <= s2):
                normal_vector[j1,:] = eig_vec_cov[:,2]/LA.norm(eig_vec_cov[:,2])
            # get attributes for local neighborhoods (eigenvalues, radius, density, curvature, ...)
            radius_kNN[j1,0]  = dist[j1,int(nn_size[j1])+0]     +1e-6             # radius of local neighborhood ?
            density[j1,0]     = 1.0*(nn_size[j1]+1) / (4.0/3.0*np.pi*np.power(radius_kNN[j1,0],3))   # local point density
            delta_Z_kNN[j1,0] = np.max(Z_vals[idx[j1,0:int(nn_size[j1])+0]]) - np.min(Z_vals[idx[j1,0:int(nn_size[j1])+0]])
            std_Z_kNN[j1,0]   = np.std(Z_vals[idx[j1,0:int(nn_size[j1])+0]])
            
            # get some 2D features
            dist_X  = X_vals[idx[j1,0:int(nn_size[j1])+1],0] - np.kron(np.ones([int(nn_size[j1])+1,1]),X_vals[idx[j1,0]])[:,0]  # ?
            dist_Y  = Y_vals[idx[j1,0:int(nn_size[j1])+1],0] - np.kron(np.ones([int(nn_size[j1])+1,1]),Y_vals[idx[j1,0]])[:,0]  # ?
            dist_2D = np.sqrt(np.power(dist_X,2) + np.power(dist_Y,2))
            radius_kNN_2D[j1,0] = np.max(dist_2D) +1e-6 
            density_2D[j1,0] = 1.0*(nn_size[j1]+1) / (np.pi * np.power(radius_kNN_2D[j1,0],2))
            # select neighboring points
            #P_2D = idx[j1,0:int(nn_size[j1])+1]
            P_2D = P
            cov_mat_2D = np.cov([X_vals[P_2D,0],Y_vals[P_2D,0]]) 
            eig_val_cov_2D, eig_vec_cov_2D = np.linalg.eig(cov_mat_2D)
            idx_eig_2D = eig_val_cov_2D.argsort()[::1] 
            eig_val_cov_2D = eig_val_cov[idx_eig_2D]
            s1_2 = eig_val_cov_2D[0]  #Min
            s2_2 = eig_val_cov_2D[1]  #Max
            EVs_2D[j1,:] = np.array([s2_2,s1_2])
            if EVs_2D[j1,1] <= 0:
                EVs_2D[j1,1] = epsilon_to_add
            if EVs_2D[j1,0] <= 0:
                EVs_2D[j1,0] = epsilon_to_add
            sum_EVs_2D[j1,0] = np.sum(EVs_2D[j1,:])  
        # normalization of eigenvalues
        EVs[:,0] = 1.0*EVs[:,0] / sum_EVs[:,0]
        EVs[:,1] = 1.0*EVs[:,1] / sum_EVs[:,0]
        EVs[:,2] = 1.0*EVs[:,2] / sum_EVs[:,0]
        EVs_2D[:,0] = 1.0*EVs_2D[:,0] / sum_EVs_2D[:,0]
        EVs_2D[:,1] = 1.0*EVs_2D[:,1] / sum_EVs_2D[:,0]
        #Now, get eigenvalue-based features by vectorized calculations:
        #1.) properties of the structure tensor according to [West et al., 2004:
        #Context-driven automated target detection in 3-D data; Toshev et al.,
        #2010: Detecting and Parsing Architecture at City Scale from Range Data;
        #Mallet et al., 2011: Relevance assessment of full-waveform lidar data
        #for urban area classification] (-> redundancy!)
        linearity = 1.0*( EVs[:,0] - EVs[:,1] ) / EVs[:,0]
        planarity = 1.0*( EVs[:,1] - EVs[:,2] ) / EVs[:,0]
        scattering = 1.0*EVs[:,2] / EVs[:,0]
        omnivariance = np.power( EVs[:,0]*EVs[:,1]*EVs[:,2], 1.0/3.0 )
        anisotropy = ( EVs[:,0] - EVs[:,2] ) / EVs[:,0]
        eigenentropy = -( EVs[:,0]*np.log(EVs[:,0]) + EVs[:,1]*np.log(EVs[:,1]) + EVs[:,2]*np.log(EVs[:,2]) )
        #2.) get surface variation, i.e. the change of curvature [Pauly et al.,
        #2003: Multi-scale feature extraction on point-sampled surfaces;
        #Rusu, 2009: Semantic 3D Object Maps for Everyday Manipulation in
        #Human Living Environments (PhD Thesis)], namely the variation of a
        #point along the surface normal (i.e. the ratio between the minimum
        #eigenvalue and the sum of all eigenvalues approximates the change
        #of curvature in a neighborhood centered around this point; note this
        #ratio is invariant under rescaling)
        change_of_curvature = 1.0*EVs[:,2] / ( EVs[:,0] + EVs[:,1] + EVs[:,2] )
        #and derive the measure of verticality [Demantke et al, 2012: Streamed 
        #Vertical Rectangle Detection in Terrestrial Laser Scans for Facade Database
        #Production]
        verticality = np.ones([point_ID_max,1])[:,0] - normal_vector[:,2]
        # Finally, get ratio of eigenvalues in 2D ...
        EV_ratio = 1.0*EVs_2D[:,1] / EVs_2D[:,0]
        # and derive features from projection to discrete image raster for ground plane
        X     = X_vals - np.min(X_vals) * np.ones([point_ID_max,1])
        Y     = Y_vals - np.min(Y_vals) * np.ones([point_ID_max,1])
        X_new = np.floor(X/raster_size) + 1
        Y_new = np.floor(Y/raster_size) + 1
        # get size of observed area
        min_X = np.min(X_new)
        max_X = np.max(X_new)
        min_Y = np.min(Y_new)
        max_Y = np.max(Y_new)
        r_acc = max_Y - min_Y + 2    #  ????
        c_acc = max_X - min_X + 2    #  ????
        # accumulate
        Acc = np.zeros([int(r_acc),int(c_acc)])
        for i in range(0,point_ID_max):
            Acc[int(Y_new[i,0]),int(X_new[i,0])] = Acc[int(Y_new[i]),int(X_new[i])] + 1
        # return a suitable vector representation
        frequency_acc_map = np.zeros([point_ID_max,1])
        h_max = np.zeros([point_ID_max,1])
        h_min = np.zeros([point_ID_max,1])
        std_z = np.zeros([point_ID_max,1])
        # use another loop  :-(  for getting accumulation map based 2D features
        for i in range(0,point_ID_max):
            bound =  np.logical_and(X_new == X_new[i],Y_new == Y_new[i]) 
            r     =  np.array(np.where(bound==True))
            r     = r[0,:]
            h_max[i,0] = np.max(Z_vals[r])
            h_min[i,0] = np.min(Z_vals[r])
            frequency_acc_map[i,0] = Acc[int(Y_new[i,0]),int(X_new[i,0])]
            std_z[i,0] = np.std(Z_vals[r])
        # height difference in the respective 2D bins (compare to cylindrical 3D neighborhood -> e.g. [Mallet et al., 2011])
        delta_z = h_max - h_min
        # Finally, combine all 3D features to a big matrix:
        attributes = np.zeros([point_ID_max,26])
        # % 8 eigenvalue-based 3D features 
        attributes[:,[0,1,2,3,4,5,6,7]] = np.array([linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, sum_EVs[:,0], change_of_curvature]).T 
        # 6 further 3D features 
        attributes[:,[8,9,10,11,12,13]] = np.array([Z_vals[:,0], radius_kNN[:,0], density[:,0], verticality, delta_Z_kNN[:,0], std_Z_kNN[:,0] ]).T
        # 7 derived 2D features                                       
        attributes[:,[14,15,16,17,18,19,20]] = np.array([radius_kNN_2D[:,0], density_2D[:,0], sum_EVs_2D[:,0], EV_ratio, frequency_acc_map[:,0], delta_z[:,0], std_z[:,0]]).T
        # (3 EVs in 3D)
        attributes[:,[21,22,23]]  = np.array([EVs])
        # (2EVs in 2D)
        attributes[:,[24,25]]  = np.array([EVs_2D]) # convert verticality to angle
        return attributes