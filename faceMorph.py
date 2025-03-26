import cv2
import numpy as np
from faceMorphTools import ImgPrepare, LandmarksFinder, AffineTrasform
import time
from sys import argv

def main():
    #carica le immagini
    img_src = cv2.imread(argv[1])
    img_tgt = cv2.imread(argv[2])

    start_time = time.time()

    #rileva i volti, centra e scala
    prep = ImgPrepare()
    img_src, face1 = prep.getPreparedImg(img_src)
    img_tgt, face2 = prep.getPreparedImg(img_tgt)

    if face1 is None:
        print("Volto nella prima foto non individuato")
        return None
    if face2 is None:
        print("Volto nella seconda foto non individuato")
        return None
    
    #individua i landmarks nelle due immagini
    src_landmarks = LandmarksFinder.findLandmarks(img_src, face1)
    tgt_landmarks = LandmarksFinder.findLandmarks(img_tgt, face2)

    #rimuove eventuali landmarks sovrapposti
    src_landmarks, tgt_landmarks = LandmarksFinder.checkLandmarks(src_landmarks, tgt_landmarks)
    
    #esegue la triangolazione sui landmarks della src
    #ritorna le coordinate dei vertici dei triangoli come indici dei landmarks e come coordinate
    src_triangles, src_triangles_coord = LandmarksFinder.Triangulation(src_landmarks, img_src)
    #utilizzando i triangoli della src individua i triangoli dell'immagine tgt
    tgt_triangles = LandmarksFinder.getTargetTriangles(src_triangles, tgt_landmarks)

    num_img = 41

    #inizializza il video
    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    video = cv2.VideoWriter("result_.avi", fourcc=fourcc, fps=8.0, frameSize=(img_src.shape[1], img_src.shape[0]))
    
    #ciclo for su t
    for t in np.linspace(0.0, 1.0, num_img):
        #calcola i triangoli intermedi in base a quelli di src e tgt, e in base a t
        intermediate_triangles = LandmarksFinder.getIntermediateTriangles(src_landmarks, tgt_landmarks, src_triangles, t)
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

    




