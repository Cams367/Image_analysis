# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:08:14 2025

@author: Camille
"""

import cv2
import os 

import pandas as pd
import re

import glob
import numpy as np
import os
from joblib import Parallel, delayed

import shutil


import time
import matplotlib.pyplot as plt
from natsort import natsorted
from PIL import Image, ImageEnhance, ImageOps

def extraire_images(nom_fichier_1,nom_fichier_2, frame_synchro1, frame_synchro2, frame_i, frame_f,nom_dossier='', frame_interval = 1): 
    """
    nom_fichier_1: nom du fichier ou se trouve le premier film 
    le second 
    
    
    la frame qui sert de reference pour le film 1 
    same pour le 2, cette étape doit être fait avant 
    
    
    
    frame i et f dans le referentiel du film 1 
    frame i va être le referentielle de numerotation
    
    
    """

    
    
    
    
    vidcap = cv2.VideoCapture(f'{nom_fichier_1}.MP4')

    frame_count=0

    
    success,image = vidcap.read()
    
    
    os.makedirs(f"{nom_dossier}{nom_fichier_1}", exist_ok=True)
    success,image = vidcap.read()
    print('Read a new frame: ', success,f"{nom_dossier}{nom_fichier_1}")

    
    while success:
        if frame_count >= frame_i and frame_count <= frame_f:
            if (frame_count-frame_i )% frame_interval == 0:  # Si la frame doit être extraite
                # Sauvegarder la frame comme fichier JPEG
                cv2.imwrite(f"{nom_dossier}{nom_fichier_1}/camera-1-%06d.jpg" % int(frame_count-frame_i), image)      # save frame as JPEG file  
                print(f"Frame {frame_count} sauvegardée.")
            # Incrémenter le compteur de frames
        frame_count += 1
        
        
        if frame_count> frame_f: 
            break
        # Lire la frame suivante
        success, image = vidcap.read()

    # Libérer les ressources
    vidcap.release()




    vidcap = cv2.VideoCapture(f'{nom_fichier_2}.MP4')
    frame_count=0

    success,image = vidcap.read()
    
    
    os.makedirs(f"{nom_dossier}{nom_fichier_2}", exist_ok=True)
    success,image = vidcap.read()
    print('Read a new frame: ', success, nom_fichier_2)

    
    
    difference_frame=frame_synchro2-frame_synchro1
    
    print(difference_frame)
    
    while success:#attention - difference_frame
        if frame_count >= (frame_i-difference_frame) and frame_count <= (frame_f-difference_frame):
            if (frame_count-(frame_i-difference_frame) ) % frame_interval == 0:  # Si la frame doit être extraite
                # Sauvegarder la frame comme fichier JPEG
                cv2.imwrite(f"{nom_dossier}{nom_fichier_2}/camera-2-%06d.jpg" % int(frame_count-(-difference_frame +frame_i)), image)      # save frame as JPEG file  
                print(f"Frame {frame_count} sauvegardée.")
            # Incrémenter le compteur de frames
        frame_count += 1
        if frame_count> (frame_f-difference_frame): 
            break
        # Lire la frame suivante
        success, image = vidcap.read()

    # Libérer les ressources
    vidcap.release()    
    
    return 




def extraitre_image_data(date, calibration,prise, frame_interval): 
    data=pd.read_excel("data_video.xlsx"  , dtype={'calibration': int})
    camera=data["camera"]
    differente_cam=list(set(camera))
    print(differente_cam)
    
    
    
    
    
    
    
    
    
    coord=np.where((data["date"]==date)&(data["calibration"]==calibration)&(data["prise"]==prise))[0]
    t1_list=-np.ones(np.size(coord))
    t2_list=-np.ones(np.size(coord))
    t_ref_list=np.zeros(np.size(coord))
    
    frame_ref_list=np.zeros(np.size(coord))
    diff_frame=np.zeros(np.size(coord))
    
    
    
    
    
    
    for i in range(np.size(coord)): 
        
        indice=coord[i]
        t1=data["t1"][indice]
    
        t2=data["t2"][indice]
    
        t_ref=data["t_ref"][indice]
        minutes, seconds = t_ref.split(':')
        sec, ms = seconds.split('.')
        total_seconds = int(minutes) * 60 + int(sec) + int(ms) / 1000
        
        t1_list[i]=t1
        t2_list[i]=t2
        f_aq=data['f_aq'][indice]
        f_calibration=data['f_aq']
        
        frame_ref=int(f_calibration[indice]*total_seconds)
        #t_ref= time.strptime(t_ref,'%M:%S.ms')
    
        t_ref_list[i]=total_seconds
        frame_ref_list[i]=frame_ref
        
        
        
        
        print(t_ref,total_seconds,frame_ref)
        
    
    #on se met dans le referentiel temporelle 
    for indice in range(np.size(frame_ref_list)-1):
    
        diff_frame[indice+1]=frame_ref_list[indice+1]-frame_ref_list[indice]
    
    
    print(t1_list)
    t1=np.max(t1_list)
    t2=np.max(t2_list)
    
    
    
    
    
    frame_i_list=np.zeros(np.size(coord))
    frame_f_list=np.zeros(np.size(coord))
    
    for indice in range(np.size(frame_ref_list)):
        
        if t1>0: 
            frame_i_list[indice]=f_calibration[indice]*t1+diff_frame[indice]
        else:
            frame_i_list[indice]=0
    
    
        if t2>0: 
            frame_f_list[indice]=f_calibration[indice]*t2+diff_frame[indice]
        else: 
            frame_f_list[indice]=-1
    
    
    print(frame_f_list, frame_i_list,t2,t1, t2_list)
    for i in range(np.size(coord)): 
        
        indice=coord[i]
        dossier_save=f"frame_extrate_for_calibration/{data['nom_video'][indice]}"
    
        frame_count=0
    
        vidcap = cv2.VideoCapture(f'{data["date"][indice]}/{data["nom_video"][indice]}.MP4')
        print(f'{data["date"][indice]}/{data["nom_video"][indice]}.MP4')
        
        frame_save_ind=[]
        frame_ref_initial=[]
        
        
        dossier_save=f"frame_extrate_for_calibration/{data['nom_video'][indice]}"
    
        success,image = vidcap.read()
    
    
        os.makedirs(dossier_save, exist_ok=True)
        success,image = vidcap.read()
        print('Read a new frame: ', success,f"{dossier_save}")
        if frame_f_list[i]>0:
    
            while success:
                if frame_count >= frame_i_list[i] and frame_count <= frame_f_list[i]:
                    verif=0
                    if (frame_count- frame_i_list[i] )% frame_interval == 0:  # Si la frame doit être extraite
                        # Sauvegarder la frame comme fichier JPEG
                        cv2.imwrite(f"{dossier_save}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[i]), image)      # save frame as JPEG file  
                        frame_save_ind.append(int(frame_count- frame_i_list[i]))
                        
                        frame_ref_initial.append(frame_count)
                        
                        
                        
                        

                        verif=verif+1
                        if verif==100: 
                            print(f"Frame {frame_count- frame_i_list[i]} sauvegardée.frame_count ")
                            verif=0
                    # Incrémenter le compteur de frames
                frame_count += 1
                
                
                if frame_count> frame_f_list[i]: #à la place de indice
                    break
                # Lire la frame suivante
                success, image = vidcap.read()
            
            # Libérer les ressources
            vidcap.release()
        else: 
            
    
            while success:
                if frame_count >= frame_i_list[i]: 
                    
                    if (frame_count- frame_i_list[i] )% frame_interval == 0:  # Si la frame doit être extraite
                        # Sauvegarder la frame comme fichier JPEG
                        cv2.imwrite(f"{dossier_save}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[i]), image)      # save frame as JPEG file  
                        print(f"Frame {frame_count} sauvegardée.{frame_count- frame_i_list[i]}")
                    # Incrémenter le compteur de frames
                frame_count += 1
                
    
                # Lire la frame suivante
                success, image = vidcap.read()
            
            # Libérer les ressources
            vidcap.release()
        # data_frame=pd.DataFrame({"numero_save":frame_save_ind, "numero_ref_initial": frame_ref_initial })
        # data_frame.to_excel(f"{dossier_save}/data_extraction.xlsx")
    return 





def extraitre_image_data_3cam(date, calibration,prise, frame_interval): 
    data=pd.read_excel("data_video.xlsx"  , dtype={'calibration': int})
    camera=data["camera"]
    differente_cam=list(set(camera))
    print(differente_cam)
    
    
    
    
    
    
    
    
    
    coord=np.where((data["date"]==date)&(data["calibration"]==calibration)&(data["prise"]==prise))[0]
    t1_list=-np.ones(np.size(coord))
    t2_list=-np.ones(np.size(coord))

    
    t_ref_list=np.zeros(np.size(coord))
    
    frame_ref_list=np.zeros(np.size(coord))
    diff_frame=np.zeros(np.size(coord))
    
    
    
    
    
    
    for i in range(np.size(coord)): 
        
        indice=coord[i]
        t1=data["t1"][indice]
    
        t2=data["t2"][indice]
    
        t_ref=data["t_ref"][indice]
        minutes, seconds = t_ref.split(':')
        sec, ms = seconds.split('.')
        total_seconds = int(minutes) * 60 + int(sec) + int(ms) / 1000
        
        t1_list[i]=t1
        t2_list[i]=t2
        f_aq=data['f_aq'][indice]
        f_calibration=data['f_aq']
        
        frame_ref=int(f_calibration[indice]*total_seconds)
        #t_ref= time.strptime(t_ref,'%M:%S.ms')
    
        t_ref_list[i]=total_seconds
        frame_ref_list[i]=frame_ref
        
        
        
        
        print(t_ref,total_seconds,frame_ref)
        
    
    #on se met dans le referentiel temporelle 
    for indice in range(np.size(frame_ref_list)-1):
    
        diff_frame[indice+1]=frame_ref_list[indice+1]-frame_ref_list[indice]
    
    
    
    t1=np.max(t1_list)
    t2=np.max(t2_list)
    
    
    
    frame_i_list=np.zeros(np.size(coord))
    frame_f_list=np.zeros(np.size(coord))
    
    for indice in range(np.size(frame_ref_list)):
        
        if t1>0: 
            frame_i_list[indice]=f_calibration[indice]*t1+diff_frame[indice]
        else:
            frame_i_list[indice]=0
    
    
        if t2>0: 
            frame_f_list[indice]=f_calibration[indice]*t2+diff_frame[indice]
        else: 
            frame_f_list[indice]=-1
    
    

    for i in range(np.size(coord)): 
        
        indice=coord[i]
        dossier_save=f"frame_extrate/{data['nom_video'][indice]}"
    
        frame_count=0
    
        vidcap = cv2.VideoCapture(f'{data["date"][indice]}/{data["nom_video"][indice]}.MP4')
        
        
        frame_save_ind=[]
        frame_ref_initial=[]
        
        
        dossier_save=f"frame_extrate/{data['nom_video'][indice]}"
    
        success,image = vidcap.read()
    
    
        os.makedirs(dossier_save, exist_ok=True)
        success,image = vidcap.read()
        print('Read a new frame: ', success,f"{dossier_save}")
        if frame_f_list[i]>0:
    
            while success:
                if frame_count >= frame_i_list[i] and frame_count <= frame_f_list[indice]:
                    verif=0
                    if (frame_count- frame_i_list[i] )% frame_interval == 0:  # Si la frame doit être extraite
                        # Sauvegarder la frame comme fichier JPEG
                        cv2.imwrite(f"{dossier_save}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[i]), image)      # save frame as JPEG file  
                        frame_save_ind.append(int(frame_count- frame_i_list[i]))
                        
                        frame_ref_initial.append(frame_count)
                        
                        
                        
                        

                        verif=verif+1
                        if verif==100: 
                            print(f"Frame {frame_count- frame_i_list[i]} sauvegardée.frame_count ")
                            verif=0
                    # Incrémenter le compteur de frames
                frame_count += 1
                
                
                if frame_count> frame_f_list[indice]: 
                    break
                # Lire la frame suivante
                success, image = vidcap.read()
            
            # Libérer les ressources
            vidcap.release()
        else: 
            
    
            while success:
                if frame_count >= frame_i_list[i]: 
                    
                    if (frame_count- frame_i_list[i] )% frame_interval == 0:  # Si la frame doit être extraite
                        # Sauvegarder la frame comme fichier JPEG
                        cv2.imwrite(f"{dossier_save}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[i]), image)      # save frame as JPEG file  
                        print(f"Frame {frame_count} sauvegardée.{frame_count- frame_i_list[i]}")
                    # Incrémenter le compteur de frames
                frame_count += 1
                
    
                # Lire la frame suivante
                success, image = vidcap.read()
            
            # Libérer les ressources
            vidcap.release()
        # data_frame=pd.DataFrame({"numero_save":frame_save_ind, "numero_ref_initial": frame_ref_initial })
        # data_frame.to_excel(f"{dossier_save}/data_extraction.xlsx")
    return 





def converte_image2(chemin1, chemin2, i, x1_1,x1_2, y1_1,y1_2, x2_1,x2_2,y2_1,y2_2, color,dossier_output,nb_digite, visualisation=False):
    imgs = {}
    
    image1=glob.glob(f"{chemin1}*%06d*"%int(i))[0]
    image2=glob.glob(f"{chemin2}*%06d*"%int(i))[0]
    
    
    imgs[1] = Image.open(image1).convert(color)
    imgs[2] = Image.open(image2).convert(color)
    
    
    
    if x1_1=='none': 
        x1_1=0
    if x1_2=='none': 
        x1_2=0            
    if y1_1=='none': 
        y1_1=0
    if y1_2=='none': 
        y1_2=0            


    if x2_1=='none': 
        x2_1=imgs[1].width
    if x2_2=='none': 
        x2_2=imgs[2].width          
    if y2_1=='none': 
        y2_1=imgs[1].height
    if y2_2=='none': 
        y2_2=imgs[2].height
                
    
    #img = img_utils.stitch_images(imgs, 2)
    img = imgs[1].crop((x1_1,y1_1,x2_1,y2_1))#[x1_1:x2_1,y1_1:y2_1]#ici on crop les images, les variables pourront être changé de manière dynamiques
    img2=imgs[2].crop((x1_2,y1_2,x2_2,y2_2))#[#x1_2:x2_2,y1_2:y2_2]
    dst = Image.new(color, (img.width + img2.width, np.max([img.height,img2.height])))#constant ici donc c'est bon
    
    dst.paste(img, (0, 0))
    #plt.imshow(dst)
    
    h_bis=int((img.height-img2.height)*0.5)
    
    dst.paste(img2, (img.width, h_bis))
    if visualisation==True: 
        print(f"{chemin1}*%06d*"%int(i),f"{chemin2}*%06d*"%int(i))
        plt.figure(0)
        plt.imshow(dst)
        
        plt.figure(1)
        plt.imshow(img)
        
        plt.figure(2)
        plt.imshow(img2)
    # img=dst
    #plt.imshow(dst)

    dst.save(os.path.join(dossier_output, f'img_{i:0{nb_digite}d}.png'), compression='lzw')
       
       
    return 




def converte_image1(chemin1, i, x1_1, y1_1, x2_1,y2_1, color,dossier_output,nb_digite, visualisation=False):
    imgs = {}
    
    image1=glob.glob(f"{chemin1}*%06d*"%int(i))[0]
    
    
    imgs[1] = Image.open(image1).convert(color)
    
    
    
    if x1_1=='none': 
        x1_1=0
         
    if y1_1=='none': 
        y1_1=0
      


    if x2_1=='none': 
        x2_1=imgs[1].width
  
    if y2_1=='none': 
        y2_1=imgs[1].height

    
    #img = img_utils.stitch_images(imgs, 2)
    img = imgs[1].crop((x1_1,y1_1,x2_1,y2_1))#[x1_1:x2_1,y1_1:y2_1]#ici on crop les images, les variables pourront être changé de manière dynamiques
    dst = Image.new(color, (img.width ,img.height))#constant ici donc c'est bon
    
    dst.paste(img, (0, 0))
    #plt.imshow(dst)
    
    
    if visualisation==True: 
        print(f"{chemin1}*%06d*"%int(i))
        plt.figure(0)
        plt.imshow(dst)
        
        plt.figure(1)
        plt.imshow(img)

    # img=dst
    #plt.imshow(dst)

    dst.save(os.path.join(dossier_output, f'img_{i:0{nb_digite}d}.png'), compression='lzw')
       
       
    return 





def extraitre_image_data_for_Antoine(date, calibration,prise, frame_interval): 
    data=pd.read_excel("data_video.xlsx"  , dtype={'calibration': int})
    camera=data["camera"]
    differente_cam=list(set(camera))
    print(differente_cam)

    
    
    coord=np.where((data["date"]==date)&(data["calibration"]==calibration)&(data["prise"]==prise))[0]
    t1_list=-np.ones(np.size(coord))
    t2_list=-np.ones(np.size(coord))
    t_ref_list=np.zeros(np.size(coord))
    
    frame_ref_list=np.zeros(np.size(coord))
    diff_frame=np.zeros(np.size(coord))
    
    
    
    
    
    
    for i in range(np.size(coord)): 
        
        indice=coord[i]
        t1=data["t1"][indice]
    
        t2=data["t2"][indice]
    
        t_ref=data["t_ref"][indice]
        minutes, seconds = t_ref.split(':')
        sec, ms = seconds.split('.')
        total_seconds = int(minutes) * 60 + int(sec) + int(ms) / 1000
        
        t1_list[i]=t1
        t2_list[i]=t2
        f_aq=data['f_aq'][indice]
        f_calibration=data['f_aq']
        
        frame_ref=int(f_calibration[indice]*total_seconds)
        #t_ref= time.strptime(t_ref,'%M:%S.ms')
    
        t_ref_list[i]=total_seconds
        frame_ref_list[i]=frame_ref
        
        
        
        
        print(t_ref,total_seconds,frame_ref)
        
    
    #on se met dans le referentiel temporelle 
    indice_ref=np.where(t1_list==max(t1_list))[0][0]
    
    print(t1_list,indice_ref)

    for indice in range(np.size(frame_ref_list)):
    
        diff_frame[indice]=frame_ref_list[indice]-frame_ref_list[indice_ref]
    
    
    
    t1=np.max(t1_list)
    t2=np.max(t2_list)
    
    print(t1,t2,diff_frame)
    
    
    
    
    frame_i_list=np.zeros(np.size(coord))
    frame_f_list=np.zeros(np.size(coord))
    
    for indice in range(np.size(frame_ref_list)):
        
        if t1>0: 
            frame_i_list[indice]=f_calibration[indice]*t1+diff_frame[indice]
        else:
            frame_i_list[indice]=0
    
    
        if t2>0: 
            frame_f_list[indice]=f_calibration[indice]*t2+diff_frame[indice]
        else: 
            frame_f_list[indice]=-1
    
    
    print("frame final",frame_f_list, frame_i_list,t2,t1, t2_list)
    for i in range(np.size(coord)): 
        
        indice=coord[i]
        dossier_save=f"frame_extrate/{data['nom_video'][indice]}"
    
        frame_count=0
    
        vidcap = cv2.VideoCapture(f'{data["date"][indice]}/{data["nom_video"][indice]}.MP4')
        print(f'{data["date"][indice]}/{data["nom_video"][indice]}.MP4')
        
        frame_save_ind=[]
        frame_ref_initial=[]
        
        
        dossier_save=f"frame_extrate/{data['camera'][indice]}/{data['date'][indice]}-cal{calibration:0{2}d}-{prise:0{2}d}"
        print(dossier_save)
        success,image = vidcap.read()
    
    
        os.makedirs(dossier_save, exist_ok=True)
        success,image = vidcap.read()
        print('Read a new frame: ', success,f"{dossier_save}")
        
        
        
        hauteur, largeur, _ = image.shape
        fps=50
        writer = cv2.VideoWriter(f"{dossier_save}/{camera[indice]}-{data['date'][indice]}-cal{calibration:0{2}d}-{prise:0{2}d}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (largeur, hauteur))
        
        
        if frame_f_list[i]>0:
    
            while success:
                if frame_count >= frame_i_list[i] and frame_count <= frame_f_list[i]:
                    verif=0
                    if (frame_count- frame_i_list[i] )% frame_interval == 0:  # Si la frame doit être extraite
                        # Sauvegarder la frame comme fichier JPEG
                        cv2.imwrite(f"{dossier_save}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[i]), image)      # save frame as JPEG file  
                        frame_save_ind.append(int(frame_count- frame_i_list[i]))
                        
                        frame_ref_initial.append(frame_count)
                        writer.write(image)
                        
                        
                        

                        verif=verif+1
                        if verif==100: 
                            print(f"Frame {frame_count- frame_i_list[i]} sauvegardée.frame_count ")
                            verif=0
                    # Incrémenter le compteur de frames
                frame_count += 1
                
                
                if frame_count> frame_f_list[i]: #à la place de indice
                    break
                # Lire la frame suivante
                success, image = vidcap.read()
            
            # Libérer les ressources
            vidcap.release()
            
            writer.release()
        else: 
            
    
            while success:
                if frame_count >= frame_i_list[i]: 
                    
                    if (frame_count- frame_i_list[i] )% frame_interval == 0:  # Si la frame doit être extraite
                        # Sauvegarder la frame comme fichier JPEG
                        cv2.imwrite(f"{dossier_save}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[i]), image)      # save frame as JPEG file  
                        print(f"Frame {frame_count} sauvegardée.{frame_count- frame_i_list[i]}")
                        writer.write(image)

                frame_count += 1
                
    
                # read next frame
                success, image = vidcap.read()
            
            #free images and video
            vidcap.release()
            writer.release()

        
        
        
    return 




def extraitre_image_data_for_Antoine_bis(date, calibration,prise, frame_interval): 
    data=pd.read_excel("data_trier/video_Patawa.xlsx"  , dtype={'calibration': int})
    camera=data["Cam"]
    
    
    print(data)
    differente_cam=list(set(camera))
    print(differente_cam)

    
    
    coord=np.where((data["date"]==date)&(data["calibration"]==calibration)&(data["prise"]==prise))[0]
    t1_list=-np.ones(np.size(coord))
    t2_list=-np.ones(np.size(coord))
    t_ref_list=np.zeros(np.size(coord))
    
    frame_ref_list=np.zeros(np.size(coord))
    diff_frame=np.zeros(np.size(coord))
    
    
    
    
    
    
    for i in range(np.size(coord)): 
        
        indice=coord[i]
        t1=data["t1"][indice]
    
        t2=data["t2"][indice]
    
        t_ref=data["t_ref"][indice]
        minutes, seconds = t_ref.split(':')
        sec, ms = seconds.split('.')
        total_seconds = int(minutes) * 60 + int(sec) + int(ms) / 1000
        
        t1_list[i]=t1
        t2_list[i]=t2
        f_aq=data['f_aq'][indice]
        f_calibration=data['f_aq']
        
        frame_ref=int(f_calibration[indice]*total_seconds)
        #t_ref= time.strptime(t_ref,'%M:%S.ms')
    
        t_ref_list[i]=total_seconds
        frame_ref_list[i]=frame_ref
        
        
        
        
        print(t_ref,total_seconds,frame_ref)
        
    
    #on se met dans le referentiel temporelle 
    indice_ref=np.where(t1_list==max(t1_list))[0][0]
    
    print(t1_list,indice_ref)

    for indice in range(np.size(frame_ref_list)):
    
        diff_frame[indice]=frame_ref_list[indice]-frame_ref_list[indice_ref]
    
    
    
    t1=np.max(t1_list)
    t2=np.max(t2_list)
    
    print(t1,t2,diff_frame)
    
    
    
    
    frame_i_list=np.zeros(np.size(coord))
    frame_f_list=np.zeros(np.size(coord))
    
    for indice in range(np.size(frame_ref_list)):
        
        if t1>0: 
            frame_i_list[indice]=f_calibration[indice]*t1+diff_frame[indice]
        else:
            frame_i_list[indice]=0
    
    
        if t2>0: 
            frame_f_list[indice]=f_calibration[indice]*t2+diff_frame[indice]
        else: 
            frame_f_list[indice]=-1
    
    
    print("frame final",frame_f_list, frame_i_list,t2,t1, t2_list)
    for i in range(np.size(coord)): 
        
        indice=coord[i]
        dossier_save=f"data/{data['date'][indice]}-cal{calibration:0{2}d}/{data['nom_video'][indice]}"
    
        frame_count=0
    
        vidcap = cv2.VideoCapture(f'{data["date"][indice]}/{data["nom_video"][indice]}.MP4')
        print(f'{data["date"][indice]}/{data["nom_video"][indice]}.MP4')
        
        frame_save_ind=[]
        frame_ref_initial=[]
        
        
        dossier_save=f"data_extrait/{data['date'][indice]}-cal{calibration:0{2}d}/{data['camera'][indice]}/{data['date'][indice]}-cal{calibration:0{2}d}-{prise:0{2}d}"
        print(dossier_save)
        success,image = vidcap.read()
    
    
        os.makedirs(dossier_save, exist_ok=True)
        success,image = vidcap.read()
        print('Read a new frame: ', success,f"{dossier_save}")
        
        
        
        hauteur, largeur, _ = image.shape
        fps=50
        writer = cv2.VideoWriter(f"{dossier_save}/{camera[indice]}-{data['date'][indice]}-cal{calibration:0{2}d}-{prise:0{2}d}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (largeur, hauteur))
        
        
        if frame_f_list[i]>0:
    
            while success:
                if frame_count >= frame_i_list[i] and frame_count <= frame_f_list[i]:
                    verif=0
                    if (frame_count- frame_i_list[i] )% frame_interval == 0:  # Si la frame doit être extraite
                        # Sauvegarder la frame comme fichier JPEG
                        cv2.imwrite(f"{dossier_save}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[i]), image)      # save frame as JPEG file  
                        frame_save_ind.append(int(frame_count- frame_i_list[i]))
                        
                        frame_ref_initial.append(frame_count)
                        writer.write(image)
                        
                        
                        

                        verif=verif+1
                        if verif==100: 
                            print(f"Frame {frame_count- frame_i_list[i]} sauvegardée.frame_count ")
                            verif=0
                    # Incrémenter le compteur de frames
                frame_count += 1
                
                
                if frame_count> frame_f_list[i]: #à la place de indice
                    break
                # Lire la frame suivante
                success, image = vidcap.read()
            
            # Libérer les ressources
            vidcap.release()
            
            writer.release()
        else: 
            
    
            while success:
                if frame_count >= frame_i_list[i]: 
                    
                    if (frame_count- frame_i_list[i] )% frame_interval == 0:  # Si la frame doit être extraite
                        # Sauvegarder la frame comme fichier JPEG
                        cv2.imwrite(f"{dossier_save}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[i]), image)      # save frame as JPEG file  
                        print(f"Frame {frame_count} sauvegardée.{frame_count- frame_i_list[i]}")
                        writer.write(image)

                    # Incrémenter le compteur de frames
                frame_count += 1
                
    
                # Lire la frame suivante
                success, image = vidcap.read()
            
            # Libérer les ressources
            vidcap.release()
            writer.release()

        # data_frame=pd.DataFrame({"numero_save":frame_save_ind, "numero_ref_initial": frame_ref_initial })
        # data_frame.to_excel(f"{dossier_save}/data_extraction.xlsx")
        
        
        
    return 

def fonction_temps_to_seconde(t_ref): 
    hour, minutes, seconds = t_ref.split(':')
    

    sec, ms = seconds.split('.')
    total_seconds = int(minutes) * 60 + int(sec) + int(ms) / 1000
    
    #warning with avidemux 
    total_seconds=total_seconds- int(sec)/1000
    
    return total_seconds


def extrait_video(date, calibration,prise, frame_interval, nb_cam=3,f_aq=240,keep_image=False,  save_folder='',data_synch="data_vol/video_Patawa-v3.xlsx"): 
    data=pd.read_excel(f"{data_synch}", dtype={'calibration': int})
    camera=data["Cam"]
    
    
    differente_cam=list(set(camera))

    
    
    coord=np.where((data["date"]==date)&(data["calibration"]==calibration)&(data["num_prise"]==prise))[0]

    t_ref_list=np.zeros(nb_cam)
    frame_ref_list=np.zeros(nb_cam)
    diff_frame=np.zeros(nb_cam)
    
    
    
    
    for i in range(nb_cam):# on part du principe que l'on étudie toujours les 3 cameras np.size(coord)): 
        indice=coord[i]
        t_ref=data["t_synch"][indice]
        t_ref_list[i]=fonction_temps_to_seconde(t_ref)
        
        
        f_calibration=f_aq
        frame_ref=round(f_calibration*fonction_temps_to_seconde(t_ref)) #round plus que int car dans le cas 3.57->4
        #t_ref= time.strptime(t_ref,'%M:%S.ms')
    
        t_ref_list[i]=fonction_temps_to_seconde(t_ref)
        frame_ref_list[i]=frame_ref
    
    print(frame_ref_list)
    for indice in range(np.size(frame_ref_list)):
    
        diff_frame[indice]=frame_ref_list[indice]-frame_ref_list[0]#on prends toujours en reference la premier camera
    
    
    coord_cam1=np.where((data["Cam"]=='Cam1')&(data["date"]==date)&(data["calibration"]==calibration)&(data["num_prise"]==prise))[0]

    nb_vol_tot=int(data["nb_vol"][coord_cam1])
    
    ti_list=np.zeros(nb_vol_tot)
    tf_list=np.zeros(nb_vol_tot)
    
    for nb_vol in range(nb_vol_tot):
        nb_vol=nb_vol+1
        
    
        
        # on part du principe que l'on étudie toujours à partir de la 1er camera
        indice=coord[0]

        t1=data[f"ti_{nb_vol}"][indice]
        
        t2=data[f"tf_{nb_vol}"][indice]
        
        
        ti_list[nb_vol-1]=fonction_temps_to_seconde(t1)
        tf_list[nb_vol-1]=fonction_temps_to_seconde(t2)


    
    
    print('list',ti_list,tf_list,diff_frame)
    
    
    for i in range(nb_cam):# on part du principe que l'on étudie toujours les 3 cameras np.size(coord)): 
        indice=coord[i]
        
        nom_video=data["nom"][indice]
        
        
        
        
        dossier_save=f"{save_folder}data_extrait/{data['date'][indice]}-cal{calibration:0{2}d}/{nom_video}"
        os.makedirs(dossier_save, exist_ok=True)

        print(dossier_save)
        vidcap = cv2.VideoCapture(f'data_vol/{data["date"][indice]}/{nom_video}.MP4')


        frame_i_list=ti_list*f_aq+diff_frame[i]
            
        frame_f_list=tf_list*f_aq+diff_frame[i]
        
        
        for k in range(np.size(frame_i_list)):
            frame_i_list[k]=int(frame_i_list[k]-1)
            frame_f_list[k]=int(frame_f_list[k]+1)        
        print(f"Cam{i+1}",frame_i_list)
        
        print(f"frame f",frame_f_list)
        
        
        success,image = vidcap.read()
        
        print('Read a new frame: ', success,f"{dossier_save}")
        frame_count=0
        frame_save_ind=[]
        frame_ref_initial=[]
        
        
        for nb_vol  in range(nb_vol_tot): 
        
            hauteur, largeur, _ = image.shape
            fps=50
            writer = cv2.VideoWriter(f"{dossier_save}/{camera[indice]}-{data['date'][indice]}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{nb_vol+1:0{2}d}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (largeur, hauteur))

            dossier_save_frame=f"{dossier_save}/{camera[indice]}-{data['date'][indice]}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{nb_vol+1:0{2}d}"
            os.makedirs(dossier_save_frame, exist_ok=True)

            print(dossier_save_frame)
            print(frame_i_list[nb_vol],frame_f_list[nb_vol])
            if not keep_image: 
                print(f'frame pas gardé dossier {dossier_save_frame}')
                
                
                
                
            if frame_f_list[nb_vol]>0:
                
                while success:
                    
                    
                    if frame_count >= frame_i_list[nb_vol] and frame_count <= frame_f_list[nb_vol]:
                        verif=0
                        if int((frame_count- int(frame_i_list[nb_vol]) ))% frame_interval == 0:  # condition for extraction 
                            # save frame as JPEG                         
                            cv2.imwrite(f"{dossier_save_frame}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[nb_vol]), image)      # save frame as JPEG file  
                            frame_save_ind.append(int(frame_count- frame_i_list[nb_vol]))
                            frame_ref_initial.append(frame_count)
                            writer.write(image)
                        
                            verif=verif+1
                            if verif==100: 
                                verif=0
                    frame_count += 1
                    
                    
                    if frame_count> frame_f_list[nb_vol]: 
                        break
                    
                    success, image = vidcap.read()# read next frame
                
                
                writer.release() # Freeing up resources
            else: 
                
        
                while success:
                    if frame_count >= frame_i_list[nb_vol]: 
                        
                        if (frame_count- frame_i_list[nb_vol] )% frame_interval == 0:  # Si la frame doit être extraite
                            # save frame as  JPEG
                            cv2.imwrite(f"{dossier_save}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[nb_vol]), image)      # save frame as JPEG file  
                            #print(f"Frame {frame_count} sauvegardée.{frame_count- frame_i_list[nb_vol]}")
                            writer.write(image)
    
                    frame_count += 1
                    
        
                    # read next frame
                    success, image = vidcap.read()
                
                # Freeing up resources
                writer.release()
                
                
            if not keep_image: 
                print(f'frame pas gardé dossier {dossier_save_frame}')
                
                shutil.rmtree(dossier_save_frame)
                

        vidcap.release()

                
        
        
    return 






def extrait_video_check_synch(date, calibration,prise, frame_interval, nb_cam=3,f_aq=240,keep_image=False,  save_folder='', keep_mp4=False,data_synch="data_vol/video_Patawa-v3.xlsx"): 
    data=pd.read_excel(f"{data_synch}", dtype={'calibration': int})
    camera=data["Cam"]
    
    
    differente_cam=list(set(camera))

    
    
    coord=np.where((data["date"]==date)&(data["calibration"]==calibration)&(data["num_prise"]==prise))[0]

    t_ref_list=np.zeros(nb_cam)
    frame_ref_list=np.zeros(nb_cam)
    diff_frame=np.zeros(nb_cam)
    
    
    
    
    for i in range(nb_cam):# on part du principe que l'on étudie toujours les 3 cameras np.size(coord)): 
        indice=coord[i]
        t_ref=data["t_synch"][indice]
        t_ref_list[i]=fonction_temps_to_seconde(t_ref)
        
        
        f_calibration=f_aq
        frame_ref=round(f_calibration*fonction_temps_to_seconde(t_ref)) #round plus que int car dans le cas 3.57->4
        #t_ref= time.strptime(t_ref,'%M:%S.ms')
    
        t_ref_list[i]=fonction_temps_to_seconde(t_ref)
        frame_ref_list[i]=frame_ref
    
    print(frame_ref_list)
    for indice in range(np.size(frame_ref_list)):
    
        diff_frame[indice]=frame_ref_list[indice]-frame_ref_list[0]#on prends toujours en reference la premier camera
    
    
    coord_cam1=np.where((data["Cam"]=='Cam1')&(data["date"]==date)&(data["calibration"]==calibration)&(data["num_prise"]==prise))[0]

    nb_vol_tot=1
    
    ti_list=np.zeros(1)
    tf_list=np.zeros(1)
    
    for nb_vol in range(nb_vol_tot):
        nb_vol=nb_vol+1
        
    
        
        # on part du principe que l'on étudie toujours à partir de la 1er camera
        indice=coord[0]

        t1=data[f"t_synch"][indice]
        
        t2=data[f"t_synch"][indice]
        
        #print('t_i',ti_list,nb_vol)
        ti_list[nb_vol-1]=fonction_temps_to_seconde(t1)
        tf_list[nb_vol-1]=fonction_temps_to_seconde(t2)


    
    
    print('list',ti_list,tf_list,diff_frame)
    
    
    for i in range(nb_cam):# on part du principe que l'on étudie toujours les 3 cameras np.size(coord)): 
        indice=coord[i]
        
        nom_video=data["nom"][indice]
        
        
        
        
        dossier_save=f"{save_folder}data_extrait_synch_check/{data['date'][indice]}-cal{calibration:0{2}d}/{nom_video}"
        os.makedirs(dossier_save, exist_ok=True)

        print(dossier_save)
        vidcap = cv2.VideoCapture(f'data_vol/{data["date"][indice]}/{nom_video}.MP4')


        frame_i_list=ti_list*f_aq+diff_frame[i]
            
        frame_f_list=tf_list*f_aq+diff_frame[i]
        
        
        for k in range(np.size(frame_i_list)):
            frame_i_list[k]=int(frame_i_list[k]-1)
            frame_f_list[k]=int(frame_f_list[k]+1)        
        print(f"Cam{i+1}",frame_i_list)
        
        print(f"frame f",frame_f_list)
        
        
        success,image = vidcap.read()
        
        print('Read a new frame: ', success,f"{dossier_save}")
        frame_count=0
        frame_save_ind=[]
        frame_ref_initial=[]
        
        
        for nb_vol  in range(nb_vol_tot): 
        
            hauteur, largeur, _ = image.shape
            fps=50
            
            
            if keep_mp4:    
                writer = cv2.VideoWriter(f"{dossier_save}/{camera[indice]}-{data['date'][indice]}-cal{calibration:0{2}d}-{prise:0{2}d}-vol{nb_vol+1:0{2}d}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (largeur, hauteur))
            # print(frame_i_list[nb_vol],frame_f_list[nb_vol])
            # print("frame controle",frame_f_list,i,frame_i_list)
            dossier_save_frame=f"{dossier_save}"
            os.makedirs(dossier_save_frame, exist_ok=True)

            print(dossier_save_frame)
            print(frame_i_list[nb_vol],frame_f_list[nb_vol])

                
                
                
                
            if frame_f_list[nb_vol]>0:
                
                
                
                while success:
                    
                    
                    #print(frame_count)
                    if frame_count >= frame_i_list[nb_vol] and frame_count <= frame_f_list[nb_vol]:
                        verif=0
                        if int((frame_count- int(frame_i_list[nb_vol]) ))% frame_interval == 0:  # Si la frame doit être extraite
                            # Sauvegarder la frame comme fichier JPEG

                            cv2.imwrite(f"{dossier_save_frame}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[nb_vol]), image)      # save frame as JPEG file  
                            frame_save_ind.append(int(frame_count- frame_i_list[nb_vol]))
                            
                            frame_ref_initial.append(frame_count)
                            
                            if keep_mp4: 
                                writer.write(image)
                            
    
                            verif=verif+1
                            if verif==100: 
                                verif=0
                        # Incrémenter le compteur de frames
                    frame_count += 1
                    
                    
                    if frame_count> frame_f_list[nb_vol]: #à la place de indice
                        break
                    # Lire la frame suivante
                    success, image = vidcap.read()
                
                # Libérer les ressources
                if keep_mp4: 
                    writer.release()
            else: 
                
        
                while success:
                    if frame_count >= frame_i_list[nb_vol]: 
                        
                        if (frame_count- frame_i_list[nb_vol] )% frame_interval == 0:  # Si la frame doit être extraite
                            # Sauvegarder la frame comme fichier JPEG
                            cv2.imwrite(f"{dossier_save}/{camera[indice]}-%06d.jpg" % int(frame_count- frame_i_list[nb_vol]), image)      # save frame as JPEG file  
                            print(f"Frame {frame_count} sauvegardée.{frame_count- frame_i_list[nb_vol]}")
                            if keep_mp4: 
                                writer.write(image)
    
                        # Incrémenter le compteur de frames
                    frame_count += 1
                    
        
                    # Lire la frame suivante
                    success, image = vidcap.read()
                
                # Libérer les ressources
                if keep_mp4: 
                    writer.release()
                
                
            if not keep_image: 
                print(f'frame pas gardé dossier {dossier_save_frame}')
                
                shutil.rmtree(dossier_save_frame)
                
                
                #os.rmdir(dossier_save_frame)
                #dossier_save_frame    
                
                
        vidcap.release()


        
    #assemblage image pour verifier calibration 
    dossier_save=f"{save_folder}data_extrait_synch_check/{data['date'][indice]}-cal{calibration:0{2}d}/{nom_video}"
    for num_image in range(3):     
        imgs = {}
        for i in range(nb_cam):# on part du principe que l'on étudie toujours les 3 cameras np.size(coord)): 
            for i in range(nb_cam):# on part du principe que l'on étudie toujours les 3 cameras np.size(coord)): 
                indice=coord[i]
                
                nom_video=data["nom"][indice]
                
                
                
                
                dossier_save=f"{save_folder}data_extrait_synch_check/{data['date'][indice]}-cal{calibration:0{2}d}/{nom_video}"
            
            
            
                nom_image=f"{dossier_save}/{camera[indice]}-%06d.jpg"%num_image
            
                imgs[i+1] = Image.open(nom_image)
            
            
            
            
                nom_video=data["nom"][indice]
        
                
        img = imgs[1]

        
        
        dst = Image.new('RGB', (imgs[1].width + imgs[2].width + imgs[3].width, np.max([imgs[1].height, imgs[2].height, imgs[3].height])))
        dst.paste(imgs[1], (0, 0))
        dst.paste(imgs[2], (imgs[1].width, 0))
        dst.paste(imgs[3], (imgs[1].width + imgs[2].width, 0))
        img=dst
        dossier_save=f"{save_folder}data_extrait_synch_check/{data['date'][indice]}-cal{calibration:0{2}d}"

        img.save(os.path.join(dossier_save, f"{data['date'][indice]}-cal{calibration:0{2}d}-{prise:0{2}d}-img_{num_image}.png"), compression='lzw')

        
        
    return 




def extraire_serie_video_check_synch(nom_fichier,frame_interval=1,save_folder='',keep_image=True,data_synch="data_vol/video_Patawa-v3.xlsx" ): 
    data_to_trate=pd.read_excel(nom_fichier)
    
    
    
    coord_cam1=np.where(data_to_trate["Cam"]=='Cam1')[0]
    
    #print(data_to_trate)

    for i in range(np.size(coord_cam1)):
        
        
        
        coord=coord_cam1[i]
        
        
        date=data_to_trate["date"][coord]

        calibration=data_to_trate["calibration"][coord]
        prise=data_to_trate["num_prise"][coord]
        
        
        

        print(date,calibration,prise)
        
        

    
        extrait_video_check_synch(date, calibration,prise, frame_interval, save_folder=save_folder,keep_image=keep_image,data_synch=data_synch)
    


    
    return 
        
        


def extraire_serie_video(nom_fichier,frame_interval=1,save_folder='',keep_image=False,data_synch="data_vol/video_Patawa-v3.xlsx"  ): 
    data_to_trate=pd.read_excel(nom_fichier)
    
    
    
    coord_cam1=np.where(data_to_trate["Cam"]=='Cam1')[0]
    
    #print(data_to_trate)

    for i in range(np.size(coord_cam1)):
        
        
        
        coord=coord_cam1[i]
        
        
        date=data_to_trate["date"][coord]

        calibration=data_to_trate["calibration"][coord]
        prise=data_to_trate["num_prise"][coord]
        
        
        

        print(date,calibration,prise)
        
        
        print("en cours",date,calibration,prise)
        
        
        
        if prise>0.5 and prise<50: 
    
            extrait_video(date, calibration,prise, frame_interval, save_folder=save_folder,keep_image=keep_image,data_synch=data_synch)



    
    return 


nom_fichier='video_treate.xlsx'        
    
extraire_serie_video(nom_fichier,save_folder='M:/')


#extraire_serie_video_check_synch(nom_fichier,save_folder='M:/')
