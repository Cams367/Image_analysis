# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:02:55 2025

@author: Camille
"""

import sys
import cv2 as cv

import os 
import glob 
import numpy as np
import matplotlib.pyplot as plt 
from skimage.metrics import structural_similarity as ssim
#%matplotlib inline
from scipy import signal, ndimage
import pandas as pd 
import scipy.ndimage

from skimage.feature import match_template
from sklearn.decomposition import PCA

from skimage.morphology import medial_axis, skeletonize



import pandas as pd



date='2025-07-23'

calibration=4
prise=1


vol=1
    
    
num_camera=1




def etude_sam_image(date, calibration, prise, vol,dossier_image="sam_treated",extention="png", save_contour=False ):
    """
    Fonction which allow to extract the the center of mass, surface, width and height from sam2 code for alll images in a csv file
    The HSV value are extract in npz file
    
    data = np.load(f'{list_dossier}.npz')
    H=data['H']
    
    main outline,     
    """
    
    nom_dossier=f'Cam{num_camera}-{date}_cal{calibration:0{2}d}_{prise:0{2}d}'
    list_dossier=os.path.join(dossier_image,f"{date}-cal{calibration:0{2}d}",f'Cam{num_camera}_{date}_cal{calibration:0{2}d}_{prise:0{2}d}',f"Cam{num_camera}-{date}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{vol:0{2}d}_segmented_objects")   
    
    
    list_image=glob.glob(f"{list_dossier}/*.{extention}")
    nb_image=np.size(list_image)
    
    
    #detail for a single image
    
    list_num_slide=[]
    
    list_cx=[]
    list_cy=[]

    list_h=[]
    list_w=[]


    list_x=[]
    list_y=[]
    
    
    list_H=[]
    list_S=[]
    list_V=[]
    
    list_surface=[]

    for num_image in range(nb_image):
        
        nom_image=list_image[num_image]
        image=cv.imread(nom_image)
        image=image [...,::-1]#to convert in rgb mode
        
        gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        mask_coord=np.where(gray!=0)
        
        
        if len(mask_coord[0])==0: 
            
            print( f"no butterflies on slide {num_image}")
            continue
        
        
        
        x_min=np.min(mask_coord[1])
        x_max=np.max(mask_coord[1])
        
        y_min=np.min(mask_coord[0])
        y_max=np.max(mask_coord[0])
        
        
        w=x_max-x_min
        h=y_max-y_min
        
        list_h.append(h)
        list_w.append(w)
        list_num_slide.append(num_image)
        
        
        #contours detection, we only keep the larger one (butterfly)
        ret, thresh = cv.threshold(gray, 0.00001, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE )
        

            
            
            
        if len(contours)>1.5: 
            list_size=np.zeros(len(contours))
            
            for k in range(len(contours)): 
                list_size[k]=len(contours[k])
            
        
        
            contour_object=contours[np.where(list_size==np.max(list_size))[0][0]]
        
        
        if len(contours)==1:
            contour_object=contours[0]
        
        
        
        #center of mass 
        
        M = cv.moments(contour_object)
        
        if M["m00"] != 0:
                 cX = int(M["m10"] / M["m00"])
                 cY = int(M["m01"] / M["m00"])
        else:
           cX, cY = 0, 0
        
        
        
        
        list_cx.append(cX)
        list_cy.append(cY)
        
        # contour is a numpy array of shape (N, 1, 2)
        # Reshape to (N, 2) for easier handling
        points = contour_object.squeeze()
        
        # Split into x and y lists
        x_list = points[:, 0]  # All x coordinates
        y_list = points[:, 1]  # All y coordinates
        
        
        # plt.imshow(thresh)
    
        # plt.scatter(x_max,y_max)
        # plt.scatter(x_min,y_min)
        
        
        
        
        





        #part about color 
        
        hsv=cv.cvtColor(image,cv.COLOR_RGB2HSV )
        
        hsv_object=hsv[np.where(gray!=0)].squeeze()
        H=hsv[np.where(gray!=0)][:,0]
        S=hsv[np.where(gray!=0)][:,1]
        V=hsv[np.where(gray!=0)][:,2]
        
        
        list_surface.append(len(np.where(gray!=0)))
        
        
        hue_bins = 361   # number of bins along Hue axis
        
        
        H_hist, _ = np.histogram(H, bins=hue_bins, range=(0, hue_bins-1))
        
        
        
        nb_colonne_SV=256
        S_hist, _ = np.histogram(S, bins=nb_colonne_SV, range=(0, hue_bins-1))
        
        
        V_hist, _ = np.histogram(V, bins=nb_colonne_SV, range=(0, hue_bins-1))

        
        
        list_H.append(H_hist)
        list_S.append(S_hist)
        list_V.append(V_hist)

        
        
        
        
        
    H = np.zeros((hue_bins,len(list_num_slide)))
    S = np.zeros((nb_colonne_SV,len(list_num_slide)))
    V = np.zeros((nb_colonne_SV,len(list_num_slide)))



    for i in range(len(list_num_slide)):
        
        
        H[:,i]=list_H[i]
        S[:,i]=list_S[i]
        V[:,i]=list_V[i]
        
        

        
        
        
    if save_contour: 
        print("chaton")
    data_save=pd.DataFrame({"num_frame":list_num_slide, "cx":list_cx, 'cy': list_cy, "h": list_h, "w": list_w, "surface":list_surface})
    data_save.to_csv(f"{list_dossier}.csv")
    
    np.savez(f"{list_dossier}.npz", H=H, S=S, V=V)

    return 



etude_sam_image(date, calibration, prise, vol)




#%%
dossier_image="sam_treated"
nom_dossier=f'Cam{num_camera}-{date}_cal{calibration:0{2}d}_{prise:0{2}d}'

list_dossier=os.path.join(dossier_image,f"{date}-cal{calibration:0{2}d}",f'Cam{num_camera}_{date}_cal{calibration:0{2}d}_{prise:0{2}d}',f"Cam{num_camera}-{date}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{vol:0{2}d}_segmented_objects")   


data = np.load(f'{list_dossier}.npz')

H=data['H']

plt.pcolormesh(np.arange(np.shape(H)[1]),np.linspace(0, np.shape(H)[0]+1, np.shape(H)[0]), H, shading="auto", cmap="viridis")

plt.xlabel('Time (slide)')
plt.ylabel("H")



#%%
#part about color 

hsv=cv.cvtColor(image,cv.COLOR_RGB2HSV )



print(np.mean(hsv[np.where(gray!=0)]))

cie=cv.cvtColor(image,cv.COLOR_RGB2Lab )



plt.imshow(thresh)

plt.scatter(x_max,y_max)
plt.scatter(x_min,y_min)

#for num_image in range(nb_image):






   
    


