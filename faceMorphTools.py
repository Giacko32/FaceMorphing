import cv2
import numpy as np
import dlib

class ImgPrepare:
    def getPreparedImg(self, img):
        face = self.detectFace(img)
        #ritorna None se non è stato individuato il volto
        if face is None:
            return (None ,None)
        return self.centerAndScale(img, face)

    def detectFace(self, img):
        #porta in scala di grigio per una migliore classificazione
        gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_detector.detectMultiScale(
            gray_scale_img, scaleFactor=1.2, minNeighbors=9
        )
        
        #prende il volto più grande identificato per evitare falsi positivi
        #se non sono individuati dei volti allora faces è una tupla vuota altrimenti è un nparray
        if type(faces) != tuple and faces.shape[0] > 0:
            faces_area = faces[:, 2] * faces[:, 3]
            biggest_face_index = np.argmax(faces_area)
            return faces[biggest_face_index]
        else:
            return None

    def centerAndScale(self, img, face):
        if face is not None:
            #trasla per far concidere il centro del volto con il centro dell'immagine
            width, height = face[2], face[3]
            img_center = (img.shape[1] // 2, img.shape[0] // 2)
            face_center = (face[0] + width // 2, face[1] + height // 2)

            tx = img_center[0] - face_center[0]
            ty = img_center[1] - face_center[1]

            M_affine = np.array([[1, 0, tx], [0, 1, ty]]).astype(np.float32)
            img = cv2.warpAffine(img, M_affine, dsize=(img.shape[1], img.shape[0]))
            #aggiorna le coordinate del volto con la traslazione
            face_coords = np.array(
                [[face[0]+tx, face[1]+ty, 1], [face[0] + width+tx, face[1] + height+ty, 1]],
                dtype=np.float32,
            ).T

            #scala in base all'altezza del volto individuata (50% dell'altezza dell'immagine)
            target_height = 0.50 * img.shape[0]
            scale = target_height / height

            M_affine = cv2.getRotationMatrix2D(img_center, 0, scale)
    
            img = cv2.warpAffine(img, M_affine, dsize=(img.shape[1], img.shape[0]))

            real_affine = np.vstack([M_affine, [0, 0, 1]])

            #trasforma le coordinate del volto in quelle scalate 
            transformed_coords = real_affine @ face_coords
            face_coords = np.round(transformed_coords[:2].T.reshape(4,)).astype(np.int32)

            return img, face_coords
        else:
            return None


class LandmarksFinder:

    @staticmethod
    def findLandmarks(img, face):
        #rileva i landmarks del volto con dlib
        face = dlib.rectangle(left=face[0], top=face[1], right=face[2], bottom=face[3])
        landmarks_detector = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )
        landmarks = landmarks_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), face)
        landmarks = np.array(
            [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)], np.float32
        )
        #aggiungo punti extra per migliore stabilità
        extra_points = [
            (0, 0),
            (img.shape[1]//2, 0),
            (0, img.shape[0]//2),
            (img.shape[1] - 1, 0),
            (0, img.shape[0] - 1),
            (img.shape[1]-1, img.shape[0]//2),
            (img.shape[1]//2, img.shape[0]-1),
            (img.shape[1] - 1, img.shape[0] - 1),
        ]
        return np.vstack([landmarks, np.array(extra_points, np.float32)])

    @staticmethod
    def checkLandmarks(src_landmarks: np.ndarray, tgt_landmarks: np.ndarray):
        # Trova indici di landmarks duplicati
        _, unique_indices = np.unique(src_landmarks, axis=0, return_index=True)
        
        # Ordina per mantenere l'ordine originale
        unique_indices = np.sort(unique_indices)

        # Filtra entrambi gli array
        return src_landmarks[unique_indices], tgt_landmarks[unique_indices]


    @staticmethod
    def Triangulation(all_landmarks, img):
        #eseguo la triangolazione di Delaunay
        subdiv = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
        for p in all_landmarks:
            subdiv.insert(tuple(p))
        triangles = subdiv.getTriangleList()
        #i triangoli hanno la shape (n, 6) quindi faccio il reshape 
        #per ottenere triplette di vertici (n, 3, 2)
        triangles = triangles.reshape(-1, 3, 2)

        #calcolo le coordinate dei triangoli come indici dei landmarks
        triangles_index = np.zeros((triangles.shape[0], 3, 1), dtype=np.int32)
        
        for t in range(triangles.shape[0]):
            for p in range(triangles[t].shape[0]):
                distances = np.linalg.norm(all_landmarks - triangles[t, p], axis=1)
                index = np.argmin(distances)  # Trova il punto più vicino
                triangles_index[t, p] = index
    
        return triangles_index, triangles

    @staticmethod
    def getTargetTriangles(src_triangles_index, tgt_landmarks):
        #trova i triangoli target con quelli corrispondenti del src
        #sfruttando gli indici dei landmarks
        return tgt_landmarks[src_triangles_index]

    @staticmethod
    def getIntermediateTriangles(src_landmarks, tgt_landmarks, src_triangles, t):
        #calcola i triangoli intermedi grazie al valore di t e i triangoli src e tgt
        return (1 - t) * src_landmarks[src_triangles] + t * tgt_landmarks[src_triangles]
        


class AffineTrasform:
    @staticmethod
    def getIntermediateImg(img_src, intermediate_triangles, src_triangles_coords):
        target_shape = (img_src.shape[0], img_src.shape[1])
        # inizializzo le map

        map_x, map_y = np.meshgrid(np.arange(0, img_src.shape[1], 1), np.arange(0, img_src.shape[0], 1))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # per ogni triangolo intermedio calcolo le coordinate corrispondenti nell'immagine iniziale
        for tr in range(intermediate_triangles.shape[0]):
            # calcolo la trasformazione affine
            startTriangle = src_triangles_coords[tr].astype(np.float32)
            finalTriangle = intermediate_triangles[tr].astype(np.float32)
            affine_src_to_int = cv2.getAffineTransform(startTriangle, finalTriangle)
            
            #inverto per fare l'inverse mapping
            inv_affine_src_to_int = np.linalg.inv(
                np.vstack([affine_src_to_int, [0, 0, 1]])
            )
            # individuo i punti interni al triangolo intermedio
            mask = np.zeros(target_shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, finalTriangle.astype(np.int32), 1)
            mask = mask.astype(np.bool_)
            # divido le coordinate (prima le colonne-x, dopo le righe-y)
            y_triangle, x_triangle = np.where(mask)
            # crea le coordinate omogenee
            omog_points = np.vstack([x_triangle, y_triangle, np.ones_like(x_triangle)])
            # ottiene i punti dell'immagine iniziale trasformando
            src_points = inv_affine_src_to_int @ omog_points
            src_x, src_y = src_points[0], src_points[1]
            # assegna le coordinate dei punti iniziali nelle mappe dell'immagine finale
            map_x[y_triangle, x_triangle] = src_x
            map_y[y_triangle, x_triangle] = src_y

        return cv2.remap(img_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)
