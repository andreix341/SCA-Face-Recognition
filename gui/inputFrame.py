import customtkinter as ctk
from tkinter import filedialog
import os
import json
import matplotlib.pyplot as plt

class InputFrame(ctk.CTkFrame):
    
    def __init__(self, master, picture_frame, staticsitcs_frame, **kwargs):
        super().__init__(master, **kwargs)
        
        self.picture_frame = picture_frame
        self.statistics_frame = staticsitcs_frame
        
        self.title = ctk.CTkLabel(self, text="Input", font=("Fira Code", 30), text_color= "blue")
        self.title.grid(row=0, column=0, columnspan=3, padx=20, pady=20)

        self.dataset_label = ctk.CTkLabel(self, text="Select Dataset Folder:")
        self.dataset_label.grid(row=1, column=0, padx=20, pady=10)
        self.dataset_button = ctk.CTkButton(self, text="Browse", command=self.browse_folder)
        self.dataset_button.grid(row=1, column=1, padx=20, pady=10)
        self.dataset_path = ctk.CTkLabel(self, text="")
        self.dataset_path.grid(row=2, column=0, columnspan=2, padx=20, pady=10)
        
        self.nr_pers_label = ctk.CTkLabel(self, text="")
        self.nr_pers_label.grid(row=3, column=0, padx=20, pady=10)
        
        self.nr_pictures_pers_label = ctk.CTkLabel(self, text="")
        self.nr_pictures_pers_label.grid(row=3, column=1, padx=20, pady=10)
        
        self.resolution_label = ctk.CTkLabel(self, text="")
        self.resolution_label.grid(row=4, column=0, columnspan=2, padx=20, pady=10)

        self.nr_pers_test_label = ctk.CTkLabel(self, text="")
        self.nr_pers_test_label.grid(row=5, column=0, padx=20, pady=10)
        self.nr_pers_test_Cb = ctk.CTkComboBox(self, values=[], state= "disabled", command=self.update_data)
        self.nr_pers_test_Cb.grid(row=5, column=1, padx=20, pady=10)
        
        self.normLabel = ctk.CTkLabel(self, text="Normalization:")
        self.normLabel.grid(row=6, column=0, padx=20, pady=10)
        self.normCb = ctk.CTkComboBox(self, values=["Manhattan", "Euclidian", "Infinity", "Cosine"], command=self.update_data)
        self.normCb.grid(row=6, column=1, padx=20, pady=10)

        self.algorithm_label = ctk.CTkLabel(self, text="Algorithm:")
        self.algorithm_label.grid(row=7, column=0, padx=20, pady=10)
        self.algorithm_Cb = ctk.CTkComboBox(self, values=["NN", "kNN", "Eigenfaces", "Eigenfaces with CR", "Lanczos"], command=self.update_k)
        self.algorithm_Cb.grid(row=7, column=1, padx=20, pady=10)
        
        self.kLabel = ctk.CTkLabel(self, text="K:")
        self.kLabel.grid(row=8, column=0, padx=20, pady=10)
        self.kEntry = ctk.CTkEntry(self)
        self.kEntry.insert(0, "1")
        self.kEntry.grid(row=8, column=1, padx=20, pady=10)
        
        self.test_picture_label = ctk.CTkLabel(self, text="Test Picture:")
        self.test_picture_label.grid(row=9, column=0, padx=20, pady=10)
        self.test_picture_button = ctk.CTkButton(self, text="Browse", command=self.browse_test_picture)
        self.test_picture_button.grid(row=9, column=1, padx=20, pady=10)
        self.test_picture_path = ctk.CTkLabel(self, text="")
        self.test_picture_path.grid(row=10, column=0, columnspan=2, padx=20, pady=10)
        
        
        self.submit_button = ctk.CTkButton(self, text="Test", command=self.test)
        self.submit_button.grid(row=11, column=0, columnspan=2, padx=20, pady=20)
        
        self.generate_statistics_button = ctk.CTkButton(self, text="Generate Statistics", command=self.generate_statistics)
        self.generate_statistics_button.grid(row=12, column=0, columnspan=2, padx=20, pady=20)
        
        self.error_label = ctk.CTkLabel(self, text="")
        self.error_label.grid(row=13, column=0, columnspan=2, padx=20, pady=20)
        

    def browse_folder(self):
        folder = filedialog.askdirectory()
        
        self.dataset_path.configure(text=folder)
        self.nr_pers_label.configure(text=f"Number of Persons: {self.get_nr_pers()}")
        self.nr_pictures_pers_label.configure(text=f"Number of Pictures: {self.get_nr_pictures_pers()}")
        self.nr_pers_test_label.configure(text="Number of Pictures Train/Test:")
        self.nr_pers_test_Cb.configure(values=self.get_nr_pers_test(), state="normal")
        self.nr_pers_test_Cb.set(self.nr_pers_test_Cb.cget("values")[-2] if self.nr_pers_test_Cb.cget("values") else "")
        self.resolution_label.configure(text=f"Resolution: {self.get_resolution()}")
        self.update_data()
 
    def browse_test_picture(self):
        folder = filedialog.askopenfilename()
        
        self.test_picture_path.configure(text=folder)
        self.update_data()
        
    def get_nr_pers(self):
        folder_path = self.dataset_path.cget("text")
        if os.path.isdir(folder_path):
            return len([name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))])
        return 0
        
    def get_nr_pictures_pers(self):
        folder_path = self.dataset_path.cget("text")
        if os.path.isdir(folder_path):
            subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
            if subfolders:
                first_folder_path = os.path.join(folder_path, subfolders[0])
            return len([file for file in os.listdir(first_folder_path) if os.path.isfile(os.path.join(first_folder_path, file))])
        return 0
    
    def get_nr_pers_test(self):
        nr_pictures = self.get_nr_pictures_pers()
        values = [f"{i}/{nr_pictures - i}" for i in range(1, nr_pictures)]
        return values
    
    def get_resolution(self):
        folder_path = self.dataset_path.cget("text")
        if os.path.isdir(folder_path):
            subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
            if subfolders:
                first_folder_path = os.path.join(folder_path, subfolders[0])
                first_picture_path = os.path.join(first_folder_path, os.listdir(first_folder_path)[0])
                picture = plt.imread(first_picture_path, 0)
                return picture.shape
        return (0, 0)
    
    def update_k(self, choice):
        if choice == "NN":
            self.kEntry.delete(0, ctk.END)
            self.kEntry.insert(0, "1")
            self.kEntry.configure(state="disabled")
        elif choice == "kNN":
            self.kEntry.delete(0, ctk.END)
            self.kEntry.insert(0, "3")
            self.kEntry.configure(state="normal")
            
        elif choice == "Eigenfaces":
            self.kEntry.delete(0, ctk.END)
            self.kEntry.insert(0, "10")
            self.kEntry.configure(state="normal")
            
        elif choice == "Eigenfaces with CR":
            self.kEntry.delete(0, ctk.END)
            self.kEntry.insert(0, "20")
            self.kEntry.configure(state="normal")
            
        elif choice == "Lanczos":
            self.kEntry.delete(0, ctk.END)
            self.kEntry.insert(0, "50")
            self.kEntry.configure(state="normal")
            
        else:
            self.kEntry.delete(0, ctk.END)
            self.kEntry.insert(0, "1")
            self.kEntry.configure(state="disabled")
            
        self.update_data()

    def split_nr_pers_test(self):
        if self.nr_pers_test_Cb.get():
            return self.nr_pers_test_Cb.get().split("/")
        return (0, 0)
    
    def test(self):
        
        if not self.dataset_path.cget("text"):
            self.error_label.configure(text="Please select a dataset folder!")
            return
        
        if not self.test_picture_path.cget("text"):
            self.error_label.configure(text="Please select a test picture!")
            return
        
        if not self.nr_pers_test_Cb.get():
            self.error_label.configure(text="Please select the number of pictures for training and testing!")
            return
        
        if not self.kEntry.get() and self.algorithm_Cb.get() == "kNN":
            self.error_label.configure(text="Please insert k!")
            return
        
        self.error_label.configure(text="")
        self.update_data()

    def update_data(self, *args):
        
        data = {
            "nr_pers": self.get_nr_pers(),
            "nr_pictures_pers": self.get_nr_pictures_pers(),
            "nr_pictures_train": self.split_nr_pers_test()[0],
            "nr_pictures_test": self.split_nr_pers_test()[1],
            "resolution": self.get_resolution(),
            "path_ds": self.dataset_path.cget("text"),
            "path_test": self.test_picture_path.cget("text"),
            "norm": self.normCb.get(),
            "algorithm": self.algorithm_Cb.get(),
            "k": int(self.kEntry.get()) if self.kEntry.get() else 1
        }
        
        with open("data.json", 'w') as json_file:
            json.dump(data, json_file, indent=4)
            
        self.picture_frame.update_content(data)
        return
    
    def generate_statistics(self):
   
        self.error_label.configure(text="")
        
        data = {
            "nr_pers": self.get_nr_pers(),
            "nr_pictures_pers": self.get_nr_pictures_pers(),
            "nr_pictures_train": self.split_nr_pers_test()[0],
            "nr_pictures_test": self.split_nr_pers_test()[1],
            "resolution": self.get_resolution(),
            "path_ds": self.dataset_path.cget("text"),
            "path_test": self.test_picture_path.cget("text"),
            "norm": self.normCb.get(),
            "algorithm": self.algorithm_Cb.get(),
            "k": int(self.kEntry.get()) if self.kEntry.get() else 1
        }
        
        file_name = f"{data['algorithm']} k={data['k']}.csv"
        if (os.path.exists(f"statistics/{file_name}")):
            self.error_label.configure(text="Statistics already generated!")
            return
        
        self.statistics_frame.generate(data)
        self.error_label.configure(text="Statistics generated!")
        
        
        