import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from logic.algorithms import Algorithms
import numpy as np

class PictureFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.algorithmLabel = ctk.CTkLabel(self, text="Algorithm: ", font=("Fira Code", 20))
        self.algorithmLabel.grid(row=0, column=0, columnspan = 2, padx=20, pady=20, sticky="nsew")
        
        self.test_picture_label = ctk.CTkLabel(self, text="Test Picture: ", font=("Fira Code", 20))
        self.test_picture_label.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        
        self.found_picture_label = ctk.CTkLabel(self, text="Found Picture: ", font=("Fira Code", 20))
        self.found_picture_label.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        
        self.test_picture_figure = plt.figure(figsize=(5, 5))
        self.test_picture_canvas = FigureCanvasTkAgg(self.test_picture_figure, self)
        self.test_picture_canvas.get_tk_widget().grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        
        self.found_picture_figure = plt.figure(figsize=(5, 5))
        self.found_picture_canvas = FigureCanvasTkAgg(self.found_picture_figure, self)
        self.found_picture_canvas.get_tk_widget().grid(row=2, column=1, padx=20, pady=20, sticky="nsew")
        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        
        
        

    def update_content(self, data):
        self.algorithmLabel.configure(text=f"Algorithm: {data['algorithm']}")
        
        try:
            self.algObj = Algorithms(data['nr_pers'], data['nr_pictures_pers'], data['nr_pictures_train'], data['resolution'], data['path_ds'], data['path_test'], data['norm'])
            A = self.algObj.createA()
            A_class_representative = self.algObj.createA(class_representative = True)
            
            self.testPicture = plt.imread(data['path_test'], 0)
            self.testPictureVect = self.testPicture.reshape((10304,))
            
            self.test_picture_figure.clear()
            self.found_picture_figure.clear()
            
            ax1 = self.test_picture_figure.add_subplot(111)
            ax1.imshow(self.testPicture, cmap='gray')
            ax1.axis('off')
            self.test_picture_canvas.draw()
            
            if data['algorithm'] == 'NN':
                i0 = self.algObj.NNAlgorithm(A, self.testPictureVect, data['norm'])
                found_picture = A[:, i0].reshape((112, 92))
                ax2 = self.found_picture_figure.add_subplot(111)
                ax2.imshow(found_picture, cmap='gray')
                ax2.axis('off')
                self.found_picture_canvas.draw()
                
            elif data['algorithm'] == 'kNN':
                i0 = self.algObj.kNNAlgorithm(A, self.testPictureVect, data['norm'], int(data['k']))
                for idx in range(int(data['k'])):
                    ax = self.found_picture_figure.add_subplot(1, int(data['k']), idx + 1)
                    found_picture = A[:, i0[idx]].reshape((112, 92))
                    ax.imshow(found_picture, cmap='gray')
                    ax.axis('off')
                self.found_picture_canvas.draw()
                
            elif data['algorithm'] == 'Eigenfaces':
                HQPB, mean, proiectie = self.algObj.EF(A, data['k'])
                
                test_image_centered = self.algObj.PICTURE_TEST.flatten() - mean
                test_image_proj = np.dot(HQPB.T, test_image_centered)
                
                i0 = self.algObj.NNAlgorithm(proiectie.T, test_image_proj, data['norm'])
                
                found_picture = A[:, i0].reshape((112, 92))
                ax2 = self.found_picture_figure.add_subplot(111)
                ax2.imshow(found_picture, cmap='gray')
                ax2.axis('off')
                self.found_picture_canvas.draw()
                
            elif data['algorithm'] == 'Eigenfaces with CR':
                HQPB, mean, proiectie = self.algObj.EF(A_class_representative, data['k'])
                
                test_image_centered = self.algObj.PICTURE_TEST.flatten() - mean
                test_image_proj = np.dot(HQPB.T, test_image_centered)
                
                i0 = self.algObj.NNAlgorithm(proiectie.T, test_image_proj, data['norm'])
                
                found_picture = A_class_representative[:, i0].reshape((112, 92))
                ax2 = self.found_picture_figure.add_subplot(111)
                ax2.imshow(found_picture, cmap='gray')
                ax2.axis('off')
                self.found_picture_canvas.draw()
                
            elif data['algorithm'] == 'Lanczos':
                
                HQPB, mean, proiectie = self.algObj.Lanczos(A, data['k'])
                i0 = self.algObj.find_lanczos(HQPB, mean, proiectie)
                
                found_picture = A[:, i0].reshape((112, 92))
                
                ax2 = self.found_picture_figure.add_subplot(111)
                ax2.imshow(found_picture, cmap='gray')
                ax2.axis('off')
                self.found_picture_canvas.draw()
            
        except FileNotFoundError as e:
            print(f"File not found:{e}")
        except PermissionError as e:
            print(f"Permission denied: {e}")
        
        