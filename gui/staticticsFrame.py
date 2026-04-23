import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from logic.algorithms import Algorithms
import numpy as np
import os
import csv

from gui.inputFrame import InputFrame
import threading

class StatisticsFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.stats_label = ctk.CTkLabel(self, text="Statistics: ", font=("Fira Code", 20))
        self.stats_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.algorithms_label = ctk.CTkLabel(self, text="Algorithm: ", font=("Fira Code", 20))
        self.algorithms_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.RR_label = ctk.CTkLabel(self, text="Recognition Rate: ", font=("Fira Code", 20))
        self.RR_label.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        
        self.AQT_label = ctk.CTkLabel(self, text="Average Query Time: ", font=("Fira Code", 20))
        self.AQT_label.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        
        self.RR_figure = plt.figure(figsize=(5, 5))
        self.RR_canvas = FigureCanvasTkAgg(self.RR_figure, self)
        self.RR_canvas.get_tk_widget().grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        
        self.AQT_figure = plt.figure(figsize=(5, 5))
        self.AQT_canvas = FigureCanvasTkAgg(self.AQT_figure, self)
        self.AQT_canvas.get_tk_widget().grid(row=2, column=1, padx=20, pady=20, sticky="nsew")

        self.file_label = ctk.CTkLabel(self, text="File: ", font=("Fira Code", 20))
        self.file_label.grid(row=3, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        
        self.buttons_frame = ctk.CTkScrollableFrame(self)
        self.buttons_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")

        self.show_buttons()
        
        self.loading_bar = ctk.CTkProgressBar(self)
        self.loading_bar.grid(row=5, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        
        self.buttons_frame.columnconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        
    def show_buttons(self):
        statistics_folder = "statistics"
        if not os.path.exists(statistics_folder):
            os.makedirs(statistics_folder)

        files = os.listdir(statistics_folder)
        for i, csv_file in enumerate(files):
            button = ctk.CTkButton(self.buttons_frame, text=csv_file.removesuffix('.csv'), command=lambda f=csv_file: self.open_csv(f))
            button.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="nsew")
            
    
    def open_csv(self, csv_file):
        self.file_label.configure(text=f"File: {csv_file}")
        self.algorithms_label.configure(text=f"Algorithm: {csv_file.split(' ')[0]}")
        
        with open(f"statistics/{csv_file}", 'r') as file:
            reader = csv.reader(file)
            data = list(reader)
            
            norms = [float(row[0]) for row in data]
            RR = [float(row[1]) for row in data]
            AQT = [float(row[2]) for row in data]
            
            self.RR_figure.clear()
            self.AQT_figure.clear()
            
            ax1 = self.RR_figure.add_subplot(111)
            ax1.plot(norms, RR, '--', marker='o')
            ax1.set_xlabel('Norm')
            ax1.set_ylabel('Recognition Rate')
            self.RR_canvas.draw()
            
            ax2 = self.AQT_figure.add_subplot(111)
            ax2.plot(norms, AQT, '--', marker='o')
            ax2.set_xlabel('Norm')
            ax2.set_ylabel('Average Query Time')
            self.AQT_canvas.draw()        

    def generate(self, data):
        
        self.loading_bar.set(0)
        self.loading_bar.start()
        
        def run_algorithm():
            try:
                
                self.algObj = Algorithms(data['nr_pers'], data['nr_pictures_pers'], data['nr_pictures_train'], data['resolution'], data['path_ds'], data['path_test'], data['norm'])
                A = self.algObj.createA()
                A_class_representative = self.algObj.createA(class_representative = True)
                
                if data['algorithm'] == 'NN' or data['algorithm'] == 'kNN':
                    for i in range(1,5):
                        RR, AQT = self.algObj.RRAQT(A, i, data['k'])
                        csv_row = [i, RR, AQT]
                        csv_name = f"statistics/{data['algorithm']} k={data['k']}.csv"
                        with open(csv_name, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(csv_row)
                            
                elif data['algorithm'] == 'Eigenfaces':
                    for i in range(1,5):
                        RR, AQT = self.algObj.RRAQT_EF(A, i, data['k'])
                        csv_row = [i, RR, AQT]
                        csv_name = f"statistics/{data['algorithm']} k={data['k']}.csv"
                        with open(csv_name, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(csv_row)
                        
                elif data['algorithm'] == 'Eigenfaces with CR':
                    for i in range(1,5):
                        RR, AQT = self.algObj.RRAQT_EF_CR(A_class_representative, i, data['k'])
                        csv_row = [i, RR, AQT]
                        csv_name = f"statistics/{data['algorithm']} k={data['k']}.csv"
                        with open(csv_name, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(csv_row)
                        
                elif data['algorithm'] == 'Lanczos':
                    for i in range(1,5):
                        RR, AQT = self.algObj.RRAQT_Lanczos(A, i, data['k'])
                        csv_row = [i, RR, AQT]
                        csv_name = f"statistics/{data['algorithm']} k={data['k']}.csv"
                        with open(csv_name, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(csv_row)
                            
                else:
                    return
                
                self.show_buttons()
            
            except Exception as e:
                print(e)
                self.loading_bar.stop()
                self.loading_bar.set(0)
                return
                
            self.loading_bar.stop()
            self.loading_bar.set(1)
            
            # self.open_csv(f"{data['algorithm']} k={data['k']}.csv")
        
        threading.Thread(target=run_algorithm).start()