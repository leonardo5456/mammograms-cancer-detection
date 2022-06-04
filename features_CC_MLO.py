# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:24:52 2022

@author: leoes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2

# there is a problem, this function hides warning
from os import environ
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ == "__main__":
    suppress_qt_warnings()

# Filter patients with 
def filter_patient(cc_filter, data, imgView):
    #Filter for CC cases
    patient = cc_filter[cc_filter["left or right breast"].str.startswith(data)] # Full Dataframe
    #patient = cc_filter[cc_filter["image view"].str.startswith(imgView)] # Full Dataframe
    patient_PI = set(list(patient["patient_id"])) # List of patients CC right or left
    #patient_PI = list(patient["patient_id"]) # List of patients CC right or left
    #pathology_PI = set(list(patient["pathology"])) #
    #pathology_PI = list(patient["pathology"])
    if data == 'R' and imgView == 'CC':
        patient_PI = [i + '_RIGHT_CC' for i in patient_PI]
    elif data == 'L' and imgView == 'CC':
        patient_PI = [i + '_LEFT_CC' for i in patient_PI]
    if data == 'R' and imgView == 'MLO':
        patient_PI = [i + '_RIGHT_MLO' for i in patient_PI]
    elif data == 'L' and imgView == 'MLO':
        patient_PI = [i + '_LEFT_MLO' for i in patient_PI]
    
    patient_PI.sort()
    #pathology_PI.sort()
    return patient_PI #, pathology_PI
   
    
def directory_read(metadata,patient_list,data ): #dataframe/ R o L / 
    #Lectura de Directorios de Imagenes Derecha
    patient_right_coincidences = patient_list[data]
    paths_cc_right = metadata[metadata["File Location"].str.contains(patient_right_coincidences)]
    files_location_cc_right = list(paths_cc_right["File Location"])
    files_location_cc_right.sort()

    full = files_location_cc_right[0] #Mamografia Orig
    roi = files_location_cc_right[1:] #ROIs
    return full,roi, patient_right_coincidences

def image_proccess(path_R_full, path_R_ROI, data):
    #Lectura de imagenes DCM
    dcm_1 = '/1-1.dcm'
    dcm_2 = '/1-2.dcm'
    
    #Lectura de imagen mamografia completa
    ImDCMFull = pydicom.dcmread(path_R_full + dcm_1)
    ImFull = np.asarray(ImDCMFull.pixel_array)
    u, v= np.shape(ImFull)
    #zeros_size = np.zeros((u,v))
    
    cont = 0
    #Lectura de ROI's
    list_data = []
    columnas = ['ID','n ROIs', 'ROI', 'Area', 'Perimetro',
                                      'Densidad', 'Compacidad', 'Contraste', 'Uniformidad']
    df = pd.DataFrame(list_data, columns=columnas)
    for path in path_R_ROI:
        
        # ROI 1-1.dcm
        ImROI_1 = pydicom.dcmread(path + dcm_1)
        ImROI_1 = np.asarray(ImROI_1.pixel_array)
        u_1, v_1 = np.shape(ImROI_1)
        # ROI 1-2.dcm
        ImROI_2 = pydicom.dcmread(path + dcm_2)
        ImROI_2 = np.asarray(ImROI_2.pixel_array)
        u_2, v_2 = np.shape(ImROI_2)
        
        #Comparamos para saber cual es la ROI util para el calculo
        if (u_1 == u) and (v_1 == v):
            ImROI = ImROI_1
        elif (u_2 == u) and (v_2 == v):
            ImROI = ImROI_2
        else:
            print('ERROR DEL PROGRAMA')
            break
        cont += 1
        ID = data+'_'+str(cont)
        #print(f'{data}_{cont}')
        print(ID)
        
        # plt.figure()
        # plt.suptitle(f'{data}_{cont}')
        # plt.imshow(ImROI, cmap = 'gray')
        # plt.axis('off')
        
        #Creación de Imagen que únicamente contiene el área de Interés con la unidad
        IdentidadROI = np.zeros((u,v)) #Mascara de zeros del tamaño de ROI
        #Cambiando 255 por 1
        for i in range(u):
            for j in range(v):
                if(ImROI[i,j]) > 0:
                    IdentidadROI[i,j] = 1 #Cambia los valores de la mascara en zeros con valores en 1
        #Multiplicando elemento con elemento
        ImagenInteres = np.multiply(ImFull,IdentidadROI)
        
        #Muestra de la región de interés de la mamografía
        # plt.figure('Region de Interes de la Mamografía')
        # plt.imshow(ImagenInteres, cmap = 'gray')
        # plt.axis('off')
        # plt.show()
        
        #Obtención de Características Relevantes
        _, ImaBin = cv2.threshold(ImagenInteres,0,255,cv2.THRESH_BINARY) #Binarización de la Imagen
        ImaBin = ImaBin.astype(np.uint8) #Conversión a Tipo de Dato UINT8 NECESARIA PARA TRABAJAR CON OPENCV
        contornos, _ = cv2.findContours(ImaBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #Calculamos los momentos 
        cnt = contornos[0]
        #M = cv2.moments(cnt)
        #Recortamos la imagen
        x,y,w,h = cv2.boundingRect(cnt) #
        ME = 50
        if x-ME <= 0 or y-ME <= 0 or w-ME <= 0 or h-ME <=0:
            ME = 30
            if x-ME <= 0 or y-ME <= 0 or w-ME <= 0 or h-ME <=0:
                ME = 0
        #imgOut = ImagenInteres[y-ME:y+h+ME, x-ME:x+h+ME]
        imgOut = ImagenInteres[y-ME:y+h+ME, x-ME:x+w+ME]
        
        #Muestra de la región de interés de la mamografía
        # plt.figure('Region de Interes de la Mamografía BOX')
        # plt.imshow(imgOut, cmap = 'gray')
        # plt.axis()
        # plt.show()
    
        #Extraccion de area
        u_Out, v_Out= np.shape(imgOut)
        area = 0
        for i in range(u_Out):
            for j in range(v_Out):
                if(imgOut[i,j]) > 0:
                    area += 1
        print(f'Area de la ROI: {area}')
        
        # Obtención del Perímetro
        '''Pixeles que constituyen el borde de la región de interés'''
        perimetro =  cv2.arcLength(cnt, True)
        print(f'Perimetro de la ROI: {perimetro}')
        
        #Densidad
        '''Relación entre el perímetro y el area'''
        densidad = perimetro ** 2 / area
        print(f'Densidad de la ROI: {densidad}')
        
        #Compacidad
        '''Relación normalizada entre el cuadrado del perímetro y el area de la región de interés'''
        compacidad = perimetro ** 2 / (4 * np.pi * area)
        print(f'Compacidad de la ROI: {compacidad}')
        
        #Contraste
        contraste  = 0
        for i in range(u_Out):
            for j in range(v_Out):
                contraste += ((i - j) ** 2) * imgOut[i,j]
        print(f'Contraste de la ROI: {contraste}')
        
        #Uniformidad
        ImagenUniformidad = imgOut
        #if ImagenUniformidad[i,j]
        uniformidad  = 0
        for i in range(u_Out):
            for j in range(v_Out):
                if ImagenUniformidad[i,j] > 0:
                    uniformidad += ImagenUniformidad[i,j] ** 2
        print(f'Uniformidad de la ROI: {uniformidad}')
    
        nROI = ''
        list_data = [(ID,nROI,cont,area,perimetro,densidad,compacidad,contraste,uniformidad)]
        new_df = pd.DataFrame(list_data, columns=columnas)
        df = df.append(new_df)
        
    # plt.figure('Mamografía Original')
    # plt.suptitle('Full Mammografia')
    # plt.imshow(ImFull, cmap = 'gray')
    # plt.axis('off')
    # plt.show()
    
    return df
    

    
'''Main'''
# Reading file "metadata" for images location
metadata = pd.read_csv('data/metadata.csv',index_col=False) 
metadata = metadata[metadata["File Location"].str.startswith('.\CBIS-DDSM\Calc-Training')]
# Reading file "data_filtered" for the cases and patology
calc_trainig_data_file = pd.read_csv('data/data_filtered.csv', index_col=False) 

#Filtro CC en dataset
#cc_filter = calc_trainig_data_file[calc_trainig_data_file["image view"].str.startswith('CC')]

pathology_data = calc_trainig_data_file[['patient_id','pathology']]

#pathology = calc_trainig_data_file['assessment'].tolist()
pathology = calc_trainig_data_file['abnormality id'].tolist()
pathology = set(pathology)


''' Filter data with steps: RIGHT-CC,  RIGHT-MLO, LEFT-CC, LEFT-MLO '''

# Patient filter - RIGHT CC
patient_right_cc_PI = filter_patient(calc_trainig_data_file,'L', 'MLO') # filtro derecha

''' DataFrame Cration '''
# Listas Vacias para el Dataframe
list_data = []
# Columnas del DataFrame
columnas = ['ID','n ROIs', 'ROI', 'Area', 'Perimetro',
                                      'Densidad', 'Compacidad', 'Contraste', 'Uniformidad' ]
df = pd.DataFrame(list_data, columns=columnas)
error_list = []
data = ''

#patient_right_cc_PI = patient_right_cc_PI[:2] # corremos solo 3 datos porque es tardado procesar todo
#patient_right_cc_PI = ['P_00631_RIGHT_CC' ,'P_00635_RIGHT_CC','P_00635']
for full in range(len(patient_right_cc_PI)):
    try:
        # Lectura de directorios
        path_R_full, path_R_ROI, data = directory_read(metadata,patient_right_cc_PI, full)
        list_data = [(data, len(path_R_ROI), '', '', '', '', '', '', '')]
        new_dataframe = pd.DataFrame(list_data, columns=columnas)
        df = df.append(new_dataframe)
            
        print(f'--------FULL IMAGE {data}--------')
        new_df = image_proccess(path_R_full, path_R_ROI, data)  # Right data
        df = df.append(new_df)
    #except Exception as e: 
    except:
        print('\n chales carnal :v \n')
        #print(repr(e))
        error_list.append(data)

        


# # Patient filter - RIGHT MLO
# patient_right_mlo_PI = filter_patient(calc_trainig_data_file,'R', 'MLO') # filtro derecha
# # Patient filter - LEFT CC
# patient_left_cc_PI = filter_patient(calc_trainig_data_file,'L', 'CC') # filtro derecha
# # Patient filter - LEFT MLO
# patient_left_mlo_PI = filter_patient(calc_trainig_data_file,'L', 'MLO') # filtro derecha


# # Listas Vacias para el Dataframe
# list_data = []
# # Columnas del DataFrame
# columnas = ['ID','n ROIs', 'ROI', 'Area', 'Perimetro',
#                                       'Densidad', 'Compacidad', 'Contraste', 'Uniformidad' ]
# df = pd.DataFrame(list_data, columns=columnas)

# patient_right_cc_PI = patient_right_cc_PI[:2] # corremos solo 3 datos porque es tardado procesar todo
# for full in range(len(patient_right_cc_PI)):
#     # Lectura de directorios
#     path_R_full, path_R_ROI, data = directory_read(metadata,patient_right_cc_PI, full)
#     list_data = [(data, len(path_R_ROI), '', '', '', '', '', '', '')]
#     new_dataframe = pd.DataFrame(list_data, columns=columnas)
#     df = df.append(new_dataframe)
    
#     print(f'--------FULL IMAGE {data}--------')
#     new_df = image_proccess(path_R_full, path_R_ROI, data)  # Right data
#     df = df.append(new_df)
    

# #Filtro pacientes izquierda
# patient_left_mlo_PI = filter_patient(calc_trainig_data_file,'L', 'CC') # filtro derecha
# #patient_right_cc_PI = filter_patient(calc_trainig_data_file,'R', 'MLO') # filtro derecha

# patient_left_mlo_PI = patient_left_mlo_PI[:1] # corremos solo 3 datos porque es tardado procesar todo
# for full in range(len(patient_left_mlo_PI)):
#     # Lectura de directorios
#     path_L_full, path_L_ROI, data = directory_read(metadata,patient_left_mlo_PI, full)
#     list_data = [(data, len(path_L_ROI), '', '', '', '', '', '', '')]
#     new_dataframe = pd.DataFrame(list_data, columns=columnas)
#     df = df.append(new_dataframe)
    
#     print(f'--------FULL IMAGE {data}--------')
#     new_df = image_proccess(path_L_full, path_L_ROI, data)  # Right data
#     df = df.append(new_df)





# Guarda datos en CSV:
df.to_csv('data/03_Caracteristicas_L_MLO.csv', header=True, index=False)




# patient_right_cc_PI = patient_right_cc[]
# training_filter = origin_file[origin_file["File Location"].str.startswith('.\CBIS-DDSM\Calc-Training')]
# cc_filter = training_filter[training_filter["File Location"].str.contains('CC')]
# full_image_filter = cc_filter[training_filter["Series Description"].str.contains('full mammogram images')]
# full_image_path = full_image_filter["File Location"]
# print(full_image_path.head(5))

