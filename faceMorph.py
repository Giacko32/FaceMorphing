import cv2
import numpy as np
from faceMorphTools import ImgPrepare, LandmarksFinder, AffineTrasform
import time
from sys import argv

def main():
    
    if len(argv) < 4:
        print("Parametri insufficenti: file1 file2 n_frame")
        return None
    
    #carica le immagini e assicura che siano in formato RGB anche se sono salvate in scala di grigi
    img_src = cv2.imread(argv[1], cv2.IMREAD_COLOR)
    img_tgt = cv2.imread(argv[2], cv2.IMREAD_COLOR)
    num_img = int(argv[3])

    if(img_src is None or img_tgt is None):
        print("File non trovato")
        return None
    
    start_time = time.time()

    img_src_area = img_src.shape[0]*img_src.shape[1]
    img_tgt_area = img_tgt.shape[0]*img_tgt.shape[1]
    
    if(img_src_area > img_tgt_area):
        img_src = cv2.resize(img_src, dsize=(img_tgt.shape[1], img_tgt.shape[0]))
    
    elif(img_tgt_area > img_src_area):
        img_tgt = cv2.resize(img_tgt, dsize=(img_src.shape[1], img_src.shape[0]))
        
    #rileva i volti, centra e scala
    prep = ImgPrepare()
    img_src = prep.getPreparedImg(img_src)
    img_tgt = prep.getPreparedImg(img_tgt)

   
    if img_src is None:
        print("Volto nella prima foto non individuato")
        return None
    if img_tgt is None:
        print("Volto nella seconda foto non individuato")
        return None
    
    #individua i landmarks nelle due immagini
    src_landmarks = LandmarksFinder.findLandmarks(img_src)
    tgt_landmarks = LandmarksFinder.findLandmarks(img_tgt)

    #rimuove eventuali landmarks sovrapposti
    src_landmarks, tgt_landmarks = LandmarksFinder.checkLandmarks(src_landmarks, tgt_landmarks)

    #esegue la triangolazione sui landmarks della src
    #ritorna le coordinate dei vertici dei triangoli come indici dei landmarks e come coordinate
    src_triangles_index, src_triangles_coord = LandmarksFinder.Triangulation(src_landmarks, img_src)
    #utilizzando i triangoli della src individua i triangoli dell'immagine tgt
    tgt_triangles = LandmarksFinder.getTargetTriangles(src_triangles_index, tgt_landmarks)

    #inizializza il video
    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    video = cv2.VideoWriter("result3.avi", fourcc=fourcc, fps=num_img/5, frameSize=(img_src.shape[1], img_src.shape[0]))
    
    #ciclo for su t
    for t in np.round(np.linspace(0.0, 1.0, num_img), 8):
        print(t)
        #calcola i triangoli intermedi in base a quelli di src e tgt, e in base a t
        intermediate_triangles = LandmarksFinder.getIntermediateTriangles(src_landmarks, tgt_landmarks, src_triangles_index, t)
        
        #trasforma l'immagine src nell'intermedia
        intermediate_img_src = AffineTrasform.getIntermediateImg(img_src, intermediate_triangles, src_triangles_coord)
       
        #trasforma l'immagine tgt nell'intermedia
        intermediate_img_tgt = AffineTrasform.getIntermediateImg(img_tgt, intermediate_triangles, tgt_triangles)

        #blending
        blended_img = cv2.addWeighted(intermediate_img_src, 1-t, intermediate_img_tgt, t, 0.0)
        
        #aggiunge il frame al video
        video.write(blended_img)
    
    video.release()
    end_time = time.time() 
    print(f"Tempo impiegato: {(end_time-start_time):.2f} secondi")

main()

    




