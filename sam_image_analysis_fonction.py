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





def trouver_sous_dossiers(dossier):
    sous_dossiers = []
    for element in os.listdir(dossier):
        chemin = os.path.join(dossier, element)
        if os.path.isdir(chemin):
            sous_dossiers.append(element)
    return sous_dossiers



def etude_sam_image(date, calibration, prise, vol,num_camera,dossier_image="sam_treated",extention="png", save_contour=False ):
    """
    Fonction which allow to extract the the center of mass, surface, width and height from sam2 code for alll images in a csv file
    The HSV value are extract in npz file
    
    data = np.load(f'{list_dossier}.npz')
    H=data['H']
    
    main outline,     
    """
    hue_bins = 361   # number of bins along Hue axis
    nb_colonne_SV=256
    nom_dossier=f'Cam{num_camera}-{date}_cal{calibration:0{2}d}_{prise:0{2}d}'
    list_dossier=os.path.join(dossier_image,f"{date}-cal{calibration:0{2}d}",f'Cam{num_camera}_{date}_cal{calibration:0{2}d}_{prise:0{2}d}',f"Cam{num_camera}-{date}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{vol:0{2}d}")   
    
    
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
        
        


        #part about color 
        
        hsv=cv.cvtColor(image,cv.COLOR_RGB2HSV )
        
        hsv_object=hsv[np.where(gray!=0)].squeeze()
        H=hsv[np.where(gray!=0)][:,0]
        S=hsv[np.where(gray!=0)][:,1]
        V=hsv[np.where(gray!=0)][:,2]
        
        
        list_surface.append(len(np.where(gray!=0)))
        
        
        
        
        
        H_hist, _ = np.histogram(H, bins=hue_bins, range=(0, hue_bins-1))
        
        
        
        
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




def etude_sam_image_z_calibration(date, calibration, prise, vol,num_camera,dossier_image="sam_treated",extention="png", save_contour=False, show_trajectory=True ):
    """
    Fonction which allow to extract the the center of mass, surface, width and height from sam2 code for alll images in a csv file
    The HSV value are extract in npz file
    
    data = np.load(f'{list_dossier}.npz')
    H=data['H']
    
    main outline,     
    """

    nom_dossier=f'Cam{num_camera}-{date}_cal{calibration:0{2}d}_{prise:0{2}d}'
    list_dossier=os.path.join(dossier_image,f"{date}-cal{calibration:0{2}d}",f'Cam{num_camera}_{date}_cal{calibration:0{2}d}_{prise:0{2}d}',f"Cam{num_camera}-{date}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{vol:0{2}d}")   
    
    
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
        
        



        
        list_surface.append(len(np.where(gray!=0)))
        
        
        
        
    if save_contour: 
        print("chaton")
    data_save=pd.DataFrame({"num_frame":list_num_slide, "cx":list_cx, 'cy': list_cy, "h": list_h, "w": list_w, "surface":list_surface})
    data_save.to_csv(f"{list_dossier}.csv")
    if show_trajectory: 
        plt.figure()
        plt.title(f"{date} cal{calibration} prise{prise}  vol{vol} cam{num_camera}")
        plt.scatter(list_cx,list_cy)

    return list_num_slide, list_cx,list_cy







def z_tracking_based_sam(date, calibration, prise=0,nb_cam=3,dossier_image="sam_treated"):
    for num_camera in range(1,nb_cam+1): 
        
        
        dossier_cal_prise=os.path.join(dossier_image,f"{date}-cal{calibration:0{2}d}",f'Cam{num_camera}_{date}_cal{calibration:0{2}d}_{prise:0{2}d}')   
    
        
        list_sous_dossiers=trouver_sous_dossiers(dossier_cal_prise)
        
        for i in range(len(list_sous_dossiers)):
            if list_sous_dossiers[i][-5:-2]=="vol": 
                vol=int(list_sous_dossiers[i][-2:])
                
                etude_sam_image_z_calibration(date, calibration, prise, vol,num_camera)
        


    

def z_commun_for_calibration(date, calibration, prise=0,nb_cam=3,dossier_image="sam_treated",save_base='data_extrait'):
    
    
    num_camera=1
    
    dossier_cal_prise=os.path.join(dossier_image,f"{date}-cal{calibration:0{2}d}",f'Cam{num_camera}_{date}_cal{calibration:0{2}d}_{prise:0{2}d}')   
    
    
    list_sous_dossiers=trouver_sous_dossiers(dossier_cal_prise)
        
    
    for i in range(len(list_sous_dossiers)):
        num_camera=1#reference camera 
        dossier_cal_prise=os.path.join(dossier_image,f"{date}-cal{calibration:0{2}d}",f'Cam{num_camera}_{date}_cal{calibration:0{2}d}_{prise:0{2}d}')   
        
        if list_sous_dossiers[i][-5:-2]=="vol": 
            vol=int(list_sous_dossiers[i][-2:])
            
            
            
            #det liste frame comm à toutes les cameras
            data_cam_ref=pd.read_csv(os.path.join(dossier_cal_prise,f"Cam{num_camera}-{date}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{vol:0{2}d}")+'.csv')
            
            
            
            frame_num_ref=data_cam_ref["num_frame"]

            common=frame_num_ref
            for num_camera in range(2,nb_cam+1):     
                dossier_cal_prise=os.path.join(dossier_image,f"{date}-cal{calibration:0{2}d}",f'Cam{num_camera}_{date}_cal{calibration:0{2}d}_{prise:0{2}d}')  
                
                
                data_cam=pd.read_csv(os.path.join(dossier_cal_prise,f"Cam{num_camera}-{date}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{vol:0{2}d}")+'.csv')
                frame_num_cam=data_cam["num_frame"]
                common = list(set(common) & set(frame_num_cam))
                
            
                
            
            num_frame_comm= common
            

            
            df=pd.DataFrame({"frame": num_frame_comm})
            
            
            for num_camera in range(1,nb_cam+1):      
                dossier_cal_prise=os.path.join(dossier_image,f"{date}-cal{calibration:0{2}d}",f'Cam{num_camera}_{date}_cal{calibration:0{2}d}_{prise:0{2}d}')  
                
                
                data_cam=pd.read_csv(os.path.join(dossier_cal_prise,f"Cam{num_camera}-{date}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{vol:0{2}d}")+'.csv')
                frame_num_cam=data_cam["num_frame"]
                cam_name=f"Cam{num_camera}"
                cx_cam=[]
                cy_cam=[]
                for i in range(len(num_frame_comm)):
                    num_frame=num_frame_comm[i]
                    
                    
                    coord=np.where(frame_num_cam==num_frame)[0][0]
                    cx_cam.append(data_cam["cx"][coord])
                    cy_cam.append(data_cam["cy"][coord])
                df[f"{cam_name}_cx"]=cx_cam
                df[f"{cam_name}_cy"]=cy_cam
            folder_save=os.path.join(save_base,f"{date}-cal{calibration:0{2}d}",f"{date}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{vol:0{2}d}")
            df.to_csv(folder_save+'.csv')




date='2025-07-29'

calibration=1
prise=0

#etude_sam_image(date, calibration, prise, vol,3)

#z_tracking_based_sam(date, calibration)


z_commun_for_calibration(date, calibration)



#%%
dossier_image="sam_treated"
nom_dossier=f'Cam{num_camera}-{date}_cal{calibration:0{2}d}_{prise:0{2}d}'

list_dossier=os.path.join(dossier_image,f"{date}-cal{calibration:0{2}d}",f'Cam{num_camera}_{date}_cal{calibration:0{2}d}_{prise:0{2}d}',f"Cam{num_camera}-{date}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{vol:0{2}d}_segmented_objects")   


data = np.load(f'{list_dossier}.npz')

H=data['H']

plt.pcolormesh(np.arange(np.shape(H)[1]),np.linspace(0, np.shape(H)[0]+1, np.shape(H)[0]), H, shading="auto", cmap="viridis")

plt.xlabel('Time (slide)')
plt.ylabel("H")







   
    


