import numpy as np
import matplotlib.pyplot as plt
import time
import statistics
from dataclasses import dataclass


@dataclass
class Algorithms:
    
    def __init__(self, nr_pers, nr_pictures_pers, nr_pictures_train, resolution, path_ds, path_test, norm):
        self.NR_PERS = int(nr_pers)
        self.NR_PICTURES_PERS = int(nr_pictures_pers)
        self.NR_PICTURES_TRAIN = int(nr_pictures_train)
        self.RESOLUTION = int(resolution[0]) * int(resolution[1])
        self.PATH_DS = path_ds
        self.PATH_TEST = path_test
        
        if norm == 'Manhattan':
            self.NORM = 1
        elif norm == 'Euclidean':
            self.NORM = 2
        elif norm == 'Infinity':
            self.NORM = 3
        elif norm == 'Cosine':
            self.NORM = 4

        if self.PATH_TEST != 'None' or self.PATH_TEST != None or self.PATH_TEST != '':
            self.PICTURE_TEST = plt.imread(self.PATH_TEST, 0)

    def createA(self, class_representative = False):
        A = np.zeros((self.RESOLUTION, self.NR_PERS * self.NR_PICTURES_TRAIN))

        for i in range(1, self.NR_PERS + 1):
            pathFolder = self.PATH_DS + r'\s' + str(i) + '//'
            
            
            if class_representative == False:
                for j in range(1, self.NR_PICTURES_TRAIN + 1):
                    pathPicture = pathFolder + str(j) + '.pgm'
                    
                    picture = plt.imread(pathPicture, 0)
                    picture_vect = np.reshape(picture, (self.RESOLUTION,))
                    
                    A[:, (i-1)*self.NR_PICTURES_TRAIN+j-1] = picture_vect
                
            else:
                random_picture = np.random.randint(1, self.NR_PICTURES_TRAIN + 1)
                pathPicture = pathFolder + str(random_picture) + '.pgm'
                picture = plt.imread(pathPicture, 0)
                picture_vect = np.reshape(picture, (self.RESOLUTION,))
                
                A[:, i-1] = picture_vect
                
        return A


    def NNAlgorithm(self, A, testPictureVect, norm) -> int:
        
        z = np.zeros((self.NR_PERS * self.NR_PICTURES_TRAIN))
        for i in range(len(z)):
            if norm == 1:
                z[i] = np.linalg.norm(testPictureVect - A[:, i], 1)
            elif norm == 2:
                z[i] = np.linalg.norm(testPictureVect - A[:, i], 2)
            elif norm == 3:
                z[i] = np.linalg.norm(testPictureVect - A[:, i], np.inf)
            else:
                z[i] = 1 - np.dot(testPictureVect, A[:, i]) / (np.linalg.norm(testPictureVect) * np.linalg.norm(A[:, i]))
        
        i0 = np.argmin(z)
        return i0

    def kNNAlgorithm(self, A, testPictureVect, norm, k) -> np.ndarray:
        
        if k == 1:
            return self.NNAlgorithm(A, testPictureVect, norm)
        
        z = np.zeros((self.NR_PERS * self.NR_PICTURES_TRAIN))
        for i in range(len(z)):
            if norm == 1:
                z[i] = np.linalg.norm(testPictureVect - A[:, i], 1)
            elif norm == 2:
                z[i] = np.linalg.norm(testPictureVect - A[:, i], 2)
            elif norm == 3:
                z[i] = np.linalg.norm(testPictureVect - A[:, i], np.inf)
            else:
                z[i] = 1 - np.dot(testPictureVect, A[:, i]) / (np.linalg.norm(testPictureVect) * np.linalg.norm(A[:, i]))
        
        ind = np.argsort(z)[:k]
        
        return ind    
        
    def RRAQT(self, A, norm, k):
        
        counter = 0
        totalTime = 0
        for i in range(1, self.NR_PERS + 1):
            pathFolder = self.PATH_DS + r'\s' + str(i) + '//'
            
            for j in range(self.NR_PICTURES_TRAIN + 1, self.NR_PICTURES_PERS + 1):
                pathPicture = pathFolder + str(j) + '.pgm'
                
                picture = plt.imread(pathPicture, 0)
                picture_vect = np.reshape(picture, (10304,))
                
                t0 = time.time()
                i0 = self.kNNAlgorithm(A, picture_vect, norm, k)
                t1 = time.time()
                
                majClass = i0 // self.NR_PICTURES_TRAIN + 1
                if k > 1:
                    p0 = statistics.mode(majClass)
                else:
                    p0 = majClass
                    
                if p0 == i:
                    counter += 1
            totalTime += t1 - t0
        
        return counter / (self.NR_PERS * (self.NR_PICTURES_PERS - self.NR_PICTURES_TRAIN)), totalTime / (self.NR_PERS * (self.NR_PICTURES_PERS - self.NR_PICTURES_TRAIN))
                    
    
    def EF(self, A, k):
        
        A_copy = np.copy(A)
        
        mean = np.mean(A_copy, axis=1) # mean on column 
        A_copy = np.transpose(np.transpose(A_copy) - mean) # data centering
        
        L = np.dot(np.transpose(A_copy), A_copy) # optimised 

        d, V = np.linalg.eig(L) # V - col as eigenvectors for L, d - vector of eigenvalues
        idx = np.argsort(d)

        idx_k = idx[-1:-k-1:-1] # keep the last k indices, corresponding to the k largest eigenvalues
        V = np.dot(A_copy, V) # V is the initial phantoms matrix
        V = V[:, idx_k] # keep the last k columns
        
        HQPB = V # this is our high quality projection basis
        proiectie = np.dot(np.transpose(A_copy), V) # projection of the data on the HQPB
        
        return HQPB, mean, proiectie
        
    def RRAQT_EF(self, A, norm, k):
        HQPB, mean, proiectie = self.EF(A, k)
        
        counter = 0
        totalTime = 0
        for i in range(1, self.NR_PERS + 1):
            pathFolder = self.PATH_DS + r'\s' + str(i) + '//'
            
            for j in range(self.NR_PICTURES_TRAIN + 1, self.NR_PICTURES_PERS + 1):
                pathPicture = pathFolder + str(j) + '.pgm'
                
                picture = plt.imread(pathPicture, 0)
                picture_centered = picture.flatten() - mean
                picture_proj = np.dot(HQPB.T, picture_centered)
                
                t0 = time.time()
                i0 = self.NNAlgorithm(proiectie.T, picture_proj, norm)
                t1 = time.time()
                
                if i0 // self.NR_PICTURES_TRAIN + 1 == i:
                    counter += 1
            totalTime += t1 - t0
        
        return counter / (self.NR_PERS * (self.NR_PICTURES_PERS - self.NR_PICTURES_TRAIN)), totalTime / (self.NR_PERS * (self.NR_PICTURES_PERS - self.NR_PICTURES_TRAIN))
    
    def RRAQT_EF_CR(self, A, norm, k):
        HQPB, mean, proiectie = self.EF(A, k)
        
        counter = 0
        totalTime = 0
        for i in range(1, self.NR_PERS + 1):
            pathFolder = self.PATH_DS + r'\s' + str(i) + '//'
            
            for j in range(self.NR_PICTURES_TRAIN + 1, self.NR_PICTURES_PERS + 1):
                pathPicture = pathFolder + str(j) + '.pgm'
                
                picture = plt.imread(pathPicture, 0)
                picture_centered = picture.flatten() - mean
                picture_proj = np.dot(HQPB.T, picture_centered)
                
                t0 = time.time()
                i0 = self.NNAlgorithm(proiectie.T, picture_proj, norm)
                t1 = time.time()
                
                if i0 + 1 == i:
                    counter += 1
            totalTime += t1 - t0
        
        return counter / (self.NR_PERS * (self.NR_PICTURES_PERS - self.NR_PICTURES_TRAIN)), totalTime / (self.NR_PERS * (self.NR_PICTURES_PERS - self.NR_PICTURES_TRAIN))
    

    def Lanczos(self, A, k):
        mean = np.mean(A, axis=1)
        A_centered = A - mean[:, np.newaxis]

        m, n = A_centered.shape
        q = np.zeros([m, k + 1])
        alpha = np.zeros(k)
        beta = np.zeros(k + 1)
        q[:, 0] = np.random.rand(m)
        q[:, 0] /= np.linalg.norm(q[:, 0])

        for i in range(k):
            w = A_centered @ (A_centered.T @ q[:, i])
            if i > 0:
                w -= beta[i] * q[:, i - 1]
            alpha[i] = np.dot(w, q[:, i])
            w -= alpha[i] * q[:, i]
            beta[i + 1] = np.linalg.norm(w)

            if beta[i + 1] == 0:
                break

            q[:, i + 1] = w / beta[i + 1]

        HQPB = q[:, :k]
        proiectie = A_centered.T @ HQPB

        return HQPB, mean, proiectie
    
    def find_lanczos(self, HQPB, mean, proiectie):
        pozaTest_centered = self.PICTURE_TEST.flatten() - mean
        proiectiePozaTest = HQPB.T @ pozaTest_centered
        i0 = self.NNAlgorithm(proiectie.T, proiectiePozaTest, self.NORM)
        return i0
    
    def RRAQT_Lanczos(self, A, norm, k):
        HQPB, mean, proiectie = self.Lanczos(A, k)
        
        counter = 0
        totalTime = 0
        
        for i in range(1, self.NR_PERS + 1):
            pathFolder = self.PATH_DS + r'\s' + str(i) + '//'
            
            for j in range(self.NR_PICTURES_TRAIN + 1, self.NR_PICTURES_PERS + 1):
                pathPicture = pathFolder + str(j) + '.pgm'
                
                picture = plt.imread(pathPicture, 0)
                picture_centered = picture.flatten() - mean
                picture_proj = np.dot(HQPB.T, picture_centered)
                
                t0 = time.time()
                i0 = self.NNAlgorithm(proiectie.T, picture_proj, norm)
                t1 = time.time()
                
                if i0 // self.NR_PICTURES_TRAIN + 1 == i:
                    counter += 1
                totalTime += t1 - t0
                
        return counter / (self.NR_PERS * (self.NR_PICTURES_PERS - self.NR_PICTURES_TRAIN)), totalTime / (self.NR_PERS * (self.NR_PICTURES_PERS - self.NR_PICTURES_TRAIN))
        
        