# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:33:32 2025

@author: Camille
"""

import numpy as np
import cv2
import glob

import pandas as pd
from joblib import Parallel, delayed
import time


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import os
import cv2.aruco as aruco
import pathlib
import cv2 as cv


from pathlib import Path



def trouver_sous_dossiers(dossier):
    sous_dossiers = []
    for element in os.listdir(dossier):
        chemin = os.path.join(dossier, element)
        if os.path.isdir(chemin):
            sous_dossiers.append(element)
    return sous_dossiers





def find_corners(nom_image,CHECKERBOARD, visualisation=False,extention='tif'):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = cv2.imread(nom_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Trouver les coins du damier
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    corners_data=0
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners_data=corners2
        np.save(f"{nom_image[:-(len(extention)+1)]}-point-calibration.npy",corners_data)
        
        if visualisation==True: 
            # Dessiner et afficher les coins
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
    else : 
        print('no', nom_image)
    return 
    






def reconstitution_matrice_calibration(chemin_list, extention_image, coint_data=(7,9),length_square=5,nb_digite=4):
    # Paramètres du damier
    CHECKERBOARD = coint_data  # Nombre de coins internes (width, height)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Préparer les points objets (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*length_square #prise en compte de la dimension de chaque carré
    
    
    for chemin in chemin_list:
        images = glob.glob(f'{chemin}*.{extention_image}')
        print(f'{chemin}*.{extention_image}')
        
        t = time.time()
        #fait la calibration de chaque image 
        results = Parallel(n_jobs=20)(delayed(find_corners)(fname, visualisation=False,extention=extention_image) for fname in images)
        
    
    
    
    
    #on s'interesse maintenant aux images dont la calibration est commune 
    
    nombre_image_detect=glob.glob(f'{chemin}*-point-calibration.npy')
    
    t1 = time.time()
    
    
    
    
    nb_image_potentielle=np.size(glob.glob(f'{chemin}*.{extention_image}'))

    liste_image_commun=[]
    
    indice_frame=np.zeros(nb_image_potentielle)
    for chemin in chemin_list:
        for nframe in range(nb_image_potentielle):
            existe_1=np.size(glob.glob(f"{chemin}*{nframe:0{nb_digite}d}-point-calibration.npy"))
            
            
            indice_frame[nframe]=indice_frame[nframe]+existe_1
        

    nb_camera=np.size(chemin_list)
    
    liste_frame=np.where(indice_frame==nb_camera)[0] #localisation des images pour lequel il y a une calibration pour les deux camera (ou plus )
    
    
    print(liste_frame)
    
    for chemin in chemin_list:
        
        objpoints = []  # Points 3d dans l'espace monde
        imgpoints = []  # Points 2d dans le plan image
        
        
        for k in range(np.size(liste_frame)): 
            nframel=liste_frame[k]
            corners2=np.load(glob.glob(f"{chemin}*{nframe:0{nb_digite}d}-point-calibration.npy")[0])
            
            
            imgpoints.append(np.array(corners2))
    
            objpoints.append(objp)
            
        t1_bis=time.time()
        print(t1-t1_bis)
        img = cv2.imread(images[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        
            
        # Calibration de la caméra
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        t2 = time.time()
        print(t2-t1_bis)
        
        np.save(f'{chemin}-matrice-calibration-comm.npy', [ret, mtx, dist, rvecs, tvecs])
 
    return 





def verification_calibration(chemin_list, extention_image, coint_data=(7,9),length_square=5,nb_digite=4,N=2):

    chemin=chemin_list[0]    

    
    
    nb_image_potentielle=np.size(glob.glob(f'{chemin}*.{extention_image}'))

    liste_image_commun=[]
    
    indice_frame=np.zeros(nb_image_potentielle)
    for chemin in chemin_list:
        for nframe in range(nb_image_potentielle):
            existe_1=np.size(glob.glob(f"{chemin}*{nframe:0{nb_digite}d}-point-calibration.npy"))
            
            
            indice_frame[nframe]=indice_frame[nframe]+existe_1
        



    nb_camera=np.size(chemin_list)
    
    liste_frame=np.where(indice_frame==nb_camera)[0]
    nb_frame_calcul=np.min([np.size(liste_frame), nb_frame_ask])
    N=np.size(liste_frame)/nb_frame_calcul
    print(liste_frame)
    list_image_frame=np.zeros(nb_frame_calcul)
    

    chemin=chemin_list[0]    
    

    objpoints = []  # Points 3d dans l'espace monde
    imgpoints = []  # Points 2d dans le plan image
    
    
    for k in range(nb_frame_calcul): 
        nframel=liste_frame[int(k*N)]
        list_image_frame[k]=nframel
        corners2=np.load(glob.glob(f"{chemin}*{nframel:0{nb_digite}d}-point-calibration.npy")[0])
        
        
        imgpoints.append(np.array(corners2))
    

    chemin=chemin_list[0]
    objpoints = []  # Points 3d dans l'espace monde
    imgpoints = []  # Points 2d dans le plan image
    
    
    for k in range(nb_frame_calcul): 
        nframel=liste_frame[int(k*N)]
        list_image_frame[k]=nframel
        corners2=np.load(glob.glob(f"{chemin}*{nframel:0{nb_digite}d}-point-calibration.npy")[0])
        
        
        imgpoints.append(np.array(corners2))
   
    
    chemin2=chemin_list[1]
    objpoints2 = []  # Points 3d dans l'espace monde
    imgpoints2 = []  # Points 2d dans le plan image
    
    
    for k in range(nb_frame_calcul): 
        nframel=liste_frame[int(k*N)]
        list_image_frame[k]=nframel
        corners2=np.load(glob.glob(f"{chemin2}*{nframel:0{nb_digite}d}-point-calibration.npy")[0])
        
        
        imgpoints2.append(np.array(corners2))
        
        
    return list_image_frame, imgpoints, imgpoints2



def reconstitution_matrice_calibration_partielle(name_cam1,name_cam2, list_name_cam,chemin_list, extention_image, coint_data=(7,9),length_square=5, nb_frame_ask=100,nb_digite=4, checkerboard_detection=True):
    # Paramètres du damier
    list_name_cam=np.array(list_name_cam)
    
    coord_c1=np.where(list_name_cam==name_cam1)[0][0]
    coord_c2=np.where(list_name_cam==name_cam2)[0][0]

    list_cam_name=[name_cam1,name_cam2]
    chemin_list=[chemin_list[coord_c1],chemin_list[coord_c2]]

    print(chemin_list)
    CHECKERBOARD = coint_data  # Nombre de coins internes (width, height)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Préparer les points objets (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*length_square
    
    for chemin in chemin_list:
        images = glob.glob(f'{chemin}*.{extention_image}')
        t = time.time()
        if checkerboard_detection: 
            results = Parallel(n_jobs=20)(delayed(find_corners)(fname,CHECKERBOARD, visualisation=False,extention=extention_image) for fname in images)

        
    nombre_image_detect=glob.glob(f'{chemin}*-point-calibration.npy')
    
    t1 = time.time()
    
    
    #for k in chemin_list
    

    nb_image_potentielle=np.size(glob.glob(f'{chemin}*.{extention_image}'))
    list_image=glob.glob(f'{chemin}*.{extention_image}')
    liste_image_commun=[]
    
    indice_frame=np.zeros(nb_image_potentielle)
    
    
    list_indice_image=np.zeros(nb_image_potentielle)
    
    for chemin in chemin_list:
        for i in range(nb_image_potentielle):
            nom_image=list_image[i]
            
            numbers = [float(s) for s in re.findall(r"-?\d+\.?\d*", nom_image)]
            
            
            
            nframe=int(-numbers[-1])


            existe_1=np.size(glob.glob(f"{chemin}*-{nframe:0{nb_digite}d}-point-calibration.npy"))
            
            indice_frame[i]=indice_frame[i]+existe_1

            
            list_indice_image[i]=nframe
        


    nb_camera=np.size(chemin_list)
    liste_frame_coord=np.where(indice_frame==nb_camera)[0]
    liste_frame=list_indice_image[liste_frame_coord] #sinon ne prends pas bien avant liste_frame était liste_frame_coord
    
    nb_frame_calcul=np.min([np.size(liste_frame), nb_frame_ask])
    N=np.size(liste_frame)/nb_frame_calcul
    list_image_frame=np.zeros(nb_frame_calcul)
    path = Path(chemin)
    parts=list(path.parts)
    parts=parts[:-2]
    folder_save_calibration_comm= os.path.join(*parts)
    
    print(folder_save_calibration_comm)
    for i in range(2):
        
        
        
        chemin=chemin_list[i]

        
        objpoints = []  # Points 3d dans l'espace monde
        imgpoints = []  # Points 2d dans le plan image
        
        
        for k in range(nb_frame_calcul): 
            nframel=int(liste_frame[int(k*N)])

            list_image_frame[k]=nframel
            corners2=np.load(glob.glob(f"{chemin}*{nframel:0{nb_digite}d}-point-calibration.npy")[0])
            
            
            imgpoints.append(np.array(corners2))
    
            objpoints.append(objp)
            
        t1_bis=time.time()
        print(t1-t1_bis)
        img = cv2.imread(images[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        
            
        # Calibration de la caméra
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        t2 = time.time()
        print(t2-t1_bis)
        
        np.savez(os.path.join(chemin, f'{name_cam1}-{name_cam2}-matrice-calibration-comm-{nb_frame_ask}.npz'), ret=ret, mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs, list_image=list_image_frame)#, dist,rvecs, tvecs])#,np.array(list_image_frame) ])

    
    data0=np.load(os.path.join(chemin, f'{name_cam1}-{name_cam2}-matrice-calibration-comm-{nb_frame_ask}.npz'))

    ret0=data0["ret"]
    mtx0=data0["mtx"]
    dist0=data0["dist"]
    rvecs0=data0["rvecs"]
    tvecs0=data0["tvecs"]
    list_image_frame0=data0["list_image"]
        
        
        
        
        
    #on vient de passer la camera -1, si on est en stereo c'est bon
    imgpoints0 = []  # Points 2d dans le plan image
    
    
    for k in range(nb_frame_calcul): 
        nframel=int(list_indice_image[int(k*N)])
        
        nframel=int(liste_frame[int(k*N)])

        #print(k,nframel )
        list_image_frame[k]=nframel
        corners2=np.load(glob.glob(f"{chemin_list[0]}*{nframel:0{nb_digite}d}-point-calibration.npy")[0])
        
        
        imgpoints0.append(np.array(corners2))        

        
        
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F=cv2.stereoCalibrate(objpoints, imgpoints0, imgpoints, mtx0, dist0, mtx, dist, gray.shape[::-1])

    R1,R2,P1,P2,Q,validPixROI1,validPixROI2= cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,gray.shape[::-1], R, T )

    np.savez(os.path.join(folder_save_calibration_comm, f'{name_cam1}-{name_cam2}-matrice-calibration-comm-{nb_frame_ask}.npz'),retval= retval, cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1, cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2, R=R, T=T, E=E, F=F,R1=R1,R2=R2,P1=P1,P2=P2,Q=Q,validPixROI1=validPixROI1,validPixROI2=validPixROI2)


    
    return 








def calibration_matrice_between_3camera(date, calibration, prise=0,CHECKERBOARD=(10,7),length_square=15*1E-3, image_extention="jpg",name_data_video='Morpho_Patawa/data_vol/video_Patawa-v3.xlsx', nb_frame_ask=100, nb_digite=6):#miss parameter ? 
    """
    
    Write for 3 camera
    Parameters
    ----------
    date : TYPE
        date of recording 
    calibration : TYPE
        number of calibration
    prise : TYPE, optional
        DESCRIPTION. The default is 0. in our experiment calibration are annoted with a prise of 0 or 50
    CHECKERBOARD : TYPE, optional
        DESCRIPTION. The default is (10,7). depend of the checkerborad square (n,k). n is the number of intersection in horizontal axis, k in vertical one  
    length_square : TYPE, optional
        DESCRIPTION. The default is 15*1E-3.
    image_extention : TYPE, optional
        DESCRIPTION. The default is "jpg".
    name_data_video : TYPE, optional
        DESCRIPTION. The default is 'Morpho_Patawa/data_vol/video_Patawa-v3.xlsx'.
    nb_frame_ask : TYPE, optional
        DESCRIPTION. The default is 100.
    nb_digite : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    None.

    """
    
    
    data=pd.read_excel(name_data_video  ,dtype={'calibration': int})
    camera=data["Cam"]
    differente_cam=list(np.sort(list(set(camera))))
    
    coord=np.where((data["date"]==date)&(data["calibration"]==calibration)&(data["num_prise"]==prise))[0]
    
    
    
    #exception for cam4, treated latter   
    number_cam=3
    
    list_chemin=[]
    
    for num_cam in range (number_cam): 
        
        cam_name=differente_cam[num_cam]
        print(cam_name)
        coord_line=np.where((data["date"]==date)&(data["calibration"]==calibration)&(data["num_prise"]==prise)&(data["Cam"]==cam_name))[0]
        
        path_extract_image=os.path.join("data_extrait", f"{date}-cal{calibration:02d}", f"{cam_name}_{date}_cal{calibration:02d}_{prise:02d}")
        
        sous_dossier=trouver_sous_dossiers(path_extract_image)
        
        for i in range(np.size(sous_dossier)): 
            path_extract_image=os.path.join(path_extract_image, sous_dossier[i])
            print( path_extract_image)
            list_chemin.append(path_extract_image+'/')
            
            list_image=glob.glob(f'{path_extract_image}/*.{image_extention}')[142]
            #find_corners(list_image,CHECKERBOARD=(10,7), visualisation=True,extention=image_extention)
    
    
    
    
    
    
    reconstitution_matrice_calibration_partielle(differente_cam[0],differente_cam[1],differente_cam,list_chemin, extention_image=image_extention, coint_data=CHECKERBOARD,length_square=length_square, nb_frame_ask=nb_frame_ask, nb_digite=nb_digite,checkerboard_detection=True)
    reconstitution_matrice_calibration_partielle(differente_cam[0],differente_cam[2],differente_cam,list_chemin, extention_image=image_extention, coint_data=CHECKERBOARD,length_square=length_square, nb_frame_ask=nb_frame_ask, nb_digite=nb_digite,checkerboard_detection=True)
    
    
    
    reconstitution_matrice_calibration_partielle(differente_cam[1],differente_cam[0],differente_cam,list_chemin, extention_image=image_extention, coint_data=CHECKERBOARD,length_square=length_square, nb_frame_ask=nb_frame_ask, nb_digite=nb_digite,checkerboard_detection=False)
    reconstitution_matrice_calibration_partielle(differente_cam[1],differente_cam[2],differente_cam,list_chemin, extention_image=image_extention, coint_data=CHECKERBOARD,length_square=length_square, nb_frame_ask=nb_frame_ask, nb_digite=nb_digite,checkerboard_detection=False)
    
    reconstitution_matrice_calibration_partielle(differente_cam[2],differente_cam[0],differente_cam,list_chemin, extention_image=image_extention, coint_data=CHECKERBOARD,length_square=length_square, nb_frame_ask=nb_frame_ask, nb_digite=nb_digite,checkerboard_detection=False)
    reconstitution_matrice_calibration_partielle(differente_cam[2],differente_cam[1],differente_cam,list_chemin, extention_image=image_extention, coint_data=CHECKERBOARD,length_square=length_square, nb_frame_ask=nb_frame_ask, nb_digite=nb_digite,checkerboard_detection=False)



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




def load_calibration(npz_path):
    data = np.load(npz_path)
    return data['cameraMatrix'], data['distCoeffs']






def safe_triangulate(P1, P2, pts1, pts2, w_thresh=1e-6):
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
    w = points_4d_hom[3]
    valid_mask = np.abs(w) > w_thresh
    
    
    
    if np.sum(valid_mask) == 0:
        raise ValueError("Tous les points triangulés sont invalides (w ≈ 0)")
    points_3d = (points_4d_hom[:3, valid_mask] / w[valid_mask]).T
    return points_3d

def rotation_matrix_from_vectors(vec1, vec2):
    """Calcule la matrice de rotation qui aligne vec1 avec vec2."""
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, -1.0):
        # Rotation de 180° autour d'un axe orthogonal à vec1
        ortho = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
        return cv2.Rodrigues(np.pi * ortho)[0]
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return R

date='2025-07-29'

calibration=1
prise=0




image_extention="jpg"




name_data_video='Morpho_Patawa/data_vol/video_Patawa-v3.xlsx'



#calibration_matrice_between_3camera(date, calibration, prise=0,CHECKERBOARD=(10,7),length_square=15*1E-3, image_extention="jpg",name_data_video='Morpho_Patawa/data_vol/video_Patawa-v3.xlsx')


#The z-measurement after SAM treatment 

#z_tracking_based_sam(date, calibration)


#z_commun_for_calibration(date, calibration)



#triangulation and adjustement


def z_alignement_stero_calibration(name_cam1,name_cam2, image_extention="jpg",name_data_video='Morpho_Patawa/data_vol/video_Patawa-v3.xlsx', nb_frame_ask=100, nb_digite=6):
    print(name_cam1,name_cam2)
    data=pd.read_excel(name_data_video  ,dtype={'calibration': int})
    coord=np.where((data["date"]==date)&(data["calibration"]==calibration)&(data["num_prise"]==prise))[0]
    
    
    path_calibration_data=os.path.join("data_extrait", f"{date}-cal{calibration:02d}")
    
    
    
    
    data_cal_matrix=np.load(path_calibration_data+f'\{name_cam1}-{name_cam2}-matrice-calibration-comm-{nb_frame_ask}.npz')
    

    # Récupération des matrices utiles
    K1 = data_cal_matrix['cameraMatrix1']
    
    D1 = data_cal_matrix['distCoeffs1']
    
    K2 = data_cal_matrix['cameraMatrix2']
    D2 = data_cal_matrix['distCoeffs2']
    
    
    
    
    R = data_cal_matrix['R']         # Rotation entre caméras
    T = data_cal_matrix['T'].reshape(3, 1)  # Translation entre caméras
    
    
    # Matrices de projection
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))        # Caméra 1
    P2 = K2 @ np.hstack((R, T))                               # Caméra 2
    
    
    
    list_chute=glob.glob(path_calibration_data+'\*vol*.csv')
    
    
    
    
    pts1=[]
    pts2=[]
    
    
    for i in range(1):#len(list_chute)): 
        data_z_vol=pd.read_csv(list_chute[i])
        
        zx_c1=data_z_vol[f'{name_cam1}_cx']    
        zy_c1=data_z_vol[f'{name_cam1}_cy']    
    
        zx_c2=data_z_vol[f'{name_cam2}_cx']    
        zy_c2=data_z_vol[f'{name_cam2}_cy']    
        
        
        for k in range(len(zx_c1)): 
            pts1.append([zx_c1[k],zy_c1[k]])
            pts2.append([zx_c2[k],zy_c2[k]])
    
    
        #plt.scatter(zx_c1,zy_c1)
    pts1=np.array(pts1)
    pts2=np.array(pts2)
    
    pts1 = pts1.T
    pts2 = pts2.T
    
    
    
    #triangulation 
    points_3d=safe_triangulate(P1, P2, pts1, pts2)
    
    # Supprimer les doublons
    unique_points = np.unique(points_3d, axis=0)#avoid point duplication and so error

    # Vérifier qu'on a assez de points
    if unique_points.shape[0] < 2:
        print("Pas assez de points uniques pour estimer la chute.")
        return np.array([np.nan, np.nan, np.nan])
    
    print(points_3d)
    
    
    
    # Calcul du vecteur de mouvement moyen, z direction
    diffs = np.diff(points_3d, axis=0)
    direction = np.mean(diffs, axis=0)
    direction /= np.linalg.norm(direction)
    
    print("Direction de la chute :", direction)
    
    z_axis=direction
    
    #☺Matrix alignement
    Z_axis = np.array([0, 0, -1])  # direction "correcte" #need to check if we want a positive or negative axis
    R_align = rotation_matrix_from_vectors(direction, Z_axis)
    
    
    R_new = R_align @ R
    T_new = R_align @ T
    
    
    print(R_align)
    # # Appliquer la rotation aux points 3D
    # points_3d_aligned = (R_align @ points_3d.T).T
    path_extract_image=os.path.join("data_extrait", f"{date}-cal{calibration:02d}", f"{name_cam1}_{date}_cal{calibration:02d}_{prise:02d}")
    
    sous_dossier=trouver_sous_dossiers(path_extract_image)
    path_image=os.path.join("data_extrait", f"{date}-cal{calibration:02d}",f"{name_cam1}_{date}_cal{calibration:02d}")
    print(sous_dossier)

    image_list=glob.glob(os.path.join(path_extract_image,sous_dossier[0])+f"\*.{image_extention}")
    image=cv.imread(image_list[0])
    
    image_size=image.shape
    
    image_size=(image_size[0],image_size[1])
    
    
    #rectification
    R1, R2, P1_rect, P2_rect, Q, _, _  = cv.stereoRectify(K1, D1, K2, D2, image_size, R_new, T_new, flags=cv.CALIB_ZERO_DISPARITY)




    #Calcul E et F
    pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
    
    
    E, _ = cv2.findEssentialMat(pts1, pts2, K1)
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    
    
    
    #save data 

    
    np.savez(os.path.join("data_extrait", f"{date}-cal{calibration:02d}") +  f'\{name_cam1}-{name_cam2}-matrice-calibration-comm-{nb_frame_ask}-align.npz',
             R=R_new, T=T_new,
             R1=R1, R2=R2, P1_rect=P1_rect, P2_rect=P2_rect, Q=Q,E=E,F=F,z_axis=z_axis, points_3d =points_3d  ,cameraMatrix1=K1,distCoeffs1=D1,cameraMatrix2=K2,distCoeffs2=D2)
    

    return 


name_data_video='Morpho_Patawa/data_vol/video_Patawa-v3.xlsx'



data=pd.read_excel(name_data_video  ,dtype={'calibration': int})
camera=data["Cam"]
differente_cam=list(np.sort(list(set(camera))))
nb_frame_ask=100
name_cam1=differente_cam[0] 
name_cam2=differente_cam[1]



z_alignement_stero_calibration(differente_cam[0],differente_cam[1])


#%%


  
def z_alignement_sterocalibration_3camera(date, calibration, prise=0, image_extention="jpg",name_data_video='Morpho_Patawa/data_vol/video_Patawa-v3.xlsx', nb_frame_ask=100):
    

        
    data=pd.read_excel(name_data_video  ,dtype={'calibration': int})
    camera=data["Cam"]
    differente_cam=list(np.sort(list(set(camera))))
    
  
     
    z_alignement_stero_calibration(differente_cam[0],differente_cam[1],image_extention=image_extention, nb_frame_ask=nb_frame_ask)
    z_alignement_stero_calibration(differente_cam[0],differente_cam[2], image_extention=image_extention, nb_frame_ask=nb_frame_ask)
    
    
    
    z_alignement_stero_calibration(differente_cam[1],differente_cam[0], image_extention=image_extention,  nb_frame_ask=nb_frame_ask)
    z_alignement_stero_calibration(differente_cam[1],differente_cam[2],image_extention=image_extention, nb_frame_ask=nb_frame_ask)
    
    z_alignement_stero_calibration(differente_cam[2],differente_cam[0],image_extention=image_extention,  nb_frame_ask=nb_frame_ask)
    z_alignement_stero_calibration(differente_cam[2],differente_cam[1], image_extention=image_extention, nb_frame_ask=nb_frame_ask)
    return 


z_alignement_sterocalibration_3camera(date,calibration)


#add controle ? 
#%%



def projection_matrix_to_dlt_column(P):
    """
    Convertit une matrice de projection 3x4 en un vecteur colonne (12x1)
    """

    P=P.reshape(-1, 1)
    

    P_list=[]
    for i in range(len(P)-1): 
        P_list.append(P[i][0])
        
        
    return P_list



def load_stero_align_2camera_12(date, calibration,name_cam1,name_cam2,  prise=0,name_data_video='Morpho_Patawa/data_vol/video_Patawa-v3.xlsx', nb_frame_ask=100): 
    
    
    
    dossier=os.path.join("data_extrait", f"{date}-cal{calibration:02d}",f"{name_cam1}-{name_cam2}-matrice-calibration-comm-{nb_frame_ask}-align.npz") 
    data=np.load(dossier )
    return data["cameraMatrix1"], data["distCoeffs1"],data["cameraMatrix2"], data["distCoeffs2"], data["R"], data["T"], data["P1_rect"], data["P2_rect"], data["R1"],data["R2"], data["Q"],data["z_axis"]
    
#K1,D1,K2,D2,R12,T12,P1,P2,R1,R2,Q,fall_axis=load_stero_align_2camera_12(date, calibration,'Cam1',"Cam2")
def final_calibration(date, calibration,camera_ref,  prise=0,name_data_video='Morpho_Patawa/data_vol/video_Patawa-v3.xlsx', nb_frame_ask=100): 
    
    
    

    if camera_ref=="Cam1": 
        K1,D1,K2,D2,R12,T12,P1,P2,R1,R2,Q12,fall_axis=load_stero_align_2camera_12(date, calibration,'Cam1',"Cam2")
        K1,D1,K3,D3,R13,T13,P1,P3,R1,R3,Q13,fall_axis=load_stero_align_2camera_12(date, calibration,'Cam1',"Cam3")
        
        dlt1 = projection_matrix_to_dlt_column(P1)
        dlt2 = projection_matrix_to_dlt_column(P2)
        dlt3 = projection_matrix_to_dlt_column(P3)

        
        
        
        
        
    
    
    if camera_ref=="Cam2": 
        K1,D1,K2,D2,R12,T12,P1,P2,R1,R2,Q12,fall_axis=load_stero_align_2camera_12(date, calibration,'Cam2',"Cam1")
        K1,D1,K3,D3,R13,T13,P1,P3,R1,R3,Q13,fall_axis=load_stero_align_2camera_12(date, calibration,'Cam2',"Cam3")
        
        dlt1 = projection_matrix_to_dlt_column(P2)
        dlt2 = projection_matrix_to_dlt_column(P1)
        dlt3 = projection_matrix_to_dlt_column(P3)
    
    if camera_ref=="Cam3": 
        K1,D1,K2,D2,R12,T12,P1,P2,R1,R2,Q12,fall_axis=load_stero_align_2camera_12(date, calibration,'Cam3',"Cam1")
        K1,D1,K3,D3,R13,T13,P1,P3,R1,R3,Q13,fall_axis=load_stero_align_2camera_12(date, calibration,'Cam3',"Cam2")
    
        dlt1 = projection_matrix_to_dlt_column(P2)
        dlt2 = projection_matrix_to_dlt_column(P3)
        dlt3 = projection_matrix_to_dlt_column(P1)    
    
    dossier_save=os.path.join("data_extrait", f"{date}-cal{calibration:02d}" +  f'tri_stero-ref{camera_ref}-{nb_frame_ask}.npz')
    print(dossier_save)
    np.savez(dossier_save,
             cameraMatrix1=K1, distCoeffs1=D1,
             cameraMatrix2=K2, distCoeffs2=D2,
             cameraMatrix3=K3, distCoeffs3=D3, 
             R_12=R12, T_12=T12,
             R_13=R13, T_13=T13,
             P1=P1, P2=P2, P3=P3,
             R1=R1, R2=R2, R3=R3,
             Q_12=Q12, Q_13=Q13,
             fall_axis=fall_axis)
    
    
    
    df=pd.DataFrame({"# cam1":dlt1,"cam2":dlt2, "cam3": dlt3 })
    
    folder_save=os.path.join("flitrak3d_data", "calib")
    
    os.makedirs(folder_save, exist_ok=True)

    df.to_csv(folder_save+f"\{date}-cal{calibration:02d}_DLTcoefs.csv", index=False)
    
    return 
final_calibration(date, calibration,"Cam1")
    
    
#%%



#%%%
nb_frame_ask=10

data_calibration=np.load(f'{chemin2}-matrice-calibration-comm-{nb_frame_ask}.npz')


list_image=data_calibration["list_image"]




nb_digite=4

for j in range(np.size(list_image)): 
    nframel=int(list_image[j])
    
    
    os.makedirs(f"Calibration/{chemin1}", exist_ok=True)
    im1=plt.imread(f"{chemin1}-img{nframel:0{nb_digite}d}.tif")
    
    
    plt.imsave(f"Calibration/{chemin1}-img{nframel:0{nb_digite}d}.jpg",im1,cmap = 'gray')
    
    im2=plt.imread(f"{chemin2}-img{nframel:0{nb_digite}d}.tif")
    
    
    plt.imsave(f"Calibration/{chemin2}-{nframel:0{nb_digite}d}.jpg",im2,cmap = 'gray')    
    
    
#%%


nb_frame_ask=10
data_stereo=np.load(f'{chemin2}-matrice-calibration-stero{nb_frame_ask}.npz')

retval=data_stereo["retval"]
cameraMatrix1=data_stereo["cameraMatrix1"]
distCoeffs1=data_stereo["distCoeffs1"]
cameraMatrix2=data_stereo["cameraMatrix2"] 
distCoeffs2=data_stereo["distCoeffs2"]
R=data_stereo["R"]
T=data_stereo["T"]
E=data_stereo["E"]
F=data_stereo["F"]



R1=data_stereo["R1"]
R2=data_stereo["R1"]
P1=data_stereo["P1"]
P2=data_stereo["P2"]

#%%


def associer_xy(x,y): 
    
    association= np.zeros((np.size(x),2))
    
    for i in range(np.size(x)):
        
        association[i][0]=x[i]
        association[i][1]=y[i]
        
    return association

point_suivit_2=pd.read_csv("Cam2-2025-03-12-cal03-01-crop-ultrafastDLC_Resnet50_cam2-gopro-crop-2Mar20shuffle2_snapshot_020.csv", skiprows=[0,2])
point_suivit_1=pd.read_csv("Cam1-2025-03-12-cal03-01-crop-ultrafastDLC_Resnet50_cam1-gopro-crop-2Mar21shuffle1_snapshot_020.csv", skiprows=[0,2])

Fuite1_x=point_suivit_1["Fuite-D"]
Fuite1_y=point_suivit_1["Fuite-D.1"]

fuite1=associer_xy(Fuite1_x,Fuite1_y)




Fuite2_x=point_suivit_2["Fuite-D"]
Fuite2_y=point_suivit_2["Fuite-D.1"]


fuite2=associer_xy(Fuite2_x,Fuite2_y)



apex1=associer_xy(point_suivit_1["Attaque-D"],point_suivit_1["Attaque-D.1"])
apex2=associer_xy(point_suivit_2["Attaque-D"],point_suivit_2["Attaque-D.1"])

base1=associer_xy(point_suivit_1["Base-D"],point_suivit_1["Base-D.1"])
base2=associer_xy(point_suivit_2["Base-D"],point_suivit_2["Base-D.1"])





#%%
# Points 2D dans les images de la baguette
points1 = np.array([[395, 306],[310,560]], dtype='float32')  # Points dans l'image de la caméra 1
points2 = np.array([[442, 960],[564,402]], dtype='float32')  # Points dans l'image de la caméra 2
#%%
# Undistort les points
fuite1_undistorted = cv2.undistortPoints(fuite1, cameraMatrix1, distCoeffs1, R=R1,P=P1)
fuite2_undistorted = cv2.undistortPoints(fuite2,cameraMatrix2, distCoeffs2, R=R2,P=P2)# R et T sont la rotation et la translation entre les caméras

apex1_undistorted= cv2.undistortPoints(apex1, cameraMatrix1, distCoeffs1, R=R1,P=P1)
apex2_undistorted= cv2.undistortPoints(apex2,cameraMatrix2, distCoeffs2, R=R2,P=P2)




base1_undistorted= cv2.undistortPoints(base1, cameraMatrix1, distCoeffs1, R=R1,P=P1)
base2_undistorted= cv2.undistortPoints(base2,cameraMatrix2, distCoeffs2, R=R2,P=P2)




def point3D(x1,y1, x2,y2,  cameraMatrix1, distCoeffs1,R1,P1,cameraMatrix2, distCoeffs2, R2,P2):
    association1=associer_xy(x1,y1)
    association2=associer_xy(x2,y2)
    
    
    point1_undistorted= cv2.undistortPoints(association1, cameraMatrix1, distCoeffs1, R=R1,P=P1)

    point2_undistorted= cv2.undistortPoints(association2,cameraMatrix2, distCoeffs2, R=R2,P=P2)


    print(P1, P2,point1_undistorted, point2_undistorted )
    
    # Triangulation des points
    points4D_hom = cv2.triangulatePoints(P1, P2, point1_undistorted, point2_undistorted)
    points3D = points4D_hom[:3] / points4D_hom[3]  # Conversion en coordonnées 3D

    print("Points 3D:\n", points3D)
    return points3D


nom_point="Attaque-D"
points3D_apex=point3D(point_suivit_1[nom_point],point_suivit_1[f'{nom_point}.1'],point_suivit_2[nom_point],point_suivit_2[f'{nom_point}.1'],cameraMatrix1, distCoeffs1,R1,P1,cameraMatrix2, distCoeffs2, R2,P2)


nom_point="Base-D"
#points3D_apex=point3D(point_suivit_1[nom_point],point_suivit_1[f'{nom_point}.1'],point_suivit_2[nom_point],point_suivit_2[f'{nom_point}.1'],cameraMatrix1, distCoeffs1,R1,P1,cameraMatrix2, distCoeffs2, R2,P2)


nom_point="Fuite-D"

#points3D_apex=point3D(point_suivit_1[nom_point],point_suivit_1[f'{nom_point}.1'],point_suivit_2[nom_point],point_suivit_2[f'{nom_point}.1'],cameraMatrix1, distCoeffs1,R1,P1,cameraMatrix2, distCoeffs2, R2,P2)
#%%
points3D_base=point3D(point_suivit_1["Base-D"],point_suivit_1["Base-D.1"],point_suivit_2["Base"],point_suivit_2["Base.1"],cameraMatrix1, distCoeffs1,R1,P1,cameraMatrix2, distCoeffs2, R2,P2)

    
points3D_fuite=point3D(point_suivit_1["Fuite"],point_suivit_1["Fuite.1"],point_suivit_2["Fuite"],point_suivit_2["Fuite.1"],cameraMatrix1, distCoeffs1,R1,P1,cameraMatrix2, distCoeffs2, R2,P2)




#%%



plt.figure(2)

plt.plot(point_suivit_1["Apex-Attaque"],point_suivit_1["Apex-Attaque.1"])
plt.plot(point_suivit_1["Base"],point_suivit_1["Base.1"])
plt.plot(point_suivit_1["Fuite"],point_suivit_1["Fuite.1"])
im=plt.imread("Phantom/2025-02-26/2/camera_1-img0000.tif")

plt.imshow(im)
#%%
# Créer une figure 3D
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

# Afficher les points 3D
#ax.scatter(points3D_apex[0], points3D_apex[1], points3D_apex[2], c='r', marker='o')



#ax.scatter(points3D_base[0], points3D_base[1], points3D_base[2], c='k', marker='o')


ax.scatter(points3D_fuite[0], points3D_fuite[1], points3D_fuite[2], c='b', marker='o')



# Configurer les axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

