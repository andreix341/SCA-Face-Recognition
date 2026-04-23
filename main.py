from gui.pictureFrame import PictureFrame
from gui.staticticsFrame import StatisticsFrame
from gui.inputFrame import InputFrame
import customtkinter as ctk

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("SCA")
        height = 800
        width = 1200
        x = self.winfo_screenwidth() // 2 - width // 2
        y = self.winfo_screenheight() // 2 - height // 2
        self.geometry("{}x{}+{}+{}".format(width, height, x, y))
        
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.pictureFrame = PictureFrame(self)
        self.statisticsFrame = StatisticsFrame(self)
        self.inputFrame = InputFrame(self, self.pictureFrame, self.statisticsFrame)
        
        self.inputFrame.grid(row=0, column=0, columnspan=1, padx=20, pady=20, sticky="nsew")
        
        self.rightFrame = ctk.CTkFrame(self)
        self.rightFrame.grid(row=0, column=1, columnspan=2, padx=20, pady=20, sticky="nsew")
        self.rightFrame.columnconfigure(0, weight=1)
        self.rightFrame.rowconfigure(2, weight=1)
        
        self.buttonFrame = ctk.CTkFrame(self.rightFrame)
        self.buttonFrame.grid(row=0, column=0, padx=20, pady=5, sticky="nsew")
        
        self.outputFrame = ctk.CTkFrame(self.rightFrame)
        self.outputFrame.grid(row=1, column=0, rowspan=2, padx=20, pady=20, sticky="nsew")
        self.outputFrame.columnconfigure(0, weight=1)
        self.outputFrame.rowconfigure(0, weight=1)
        
        self.buttonPicture = ctk.CTkButton(self.buttonFrame, text="Picture Frame", command=self.show_picture_frame)
        self.buttonPicture.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.buttonStatistics = ctk.CTkButton(self.buttonFrame, text="Statistics Frame", command=self.show_statistics_frame)
        self.buttonStatistics.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.buttonFrame.columnconfigure(0, weight=1)
        self.buttonFrame.columnconfigure(1, weight=1)
        self.buttonFrame.rowconfigure(0, weight=1)
        
        self.pictureFrame.grid(row=0, column=0, sticky="nsew", in_=self.outputFrame)
        self.statisticsFrame.grid(row=0, column=0, sticky="nsew", in_=self.outputFrame)
        
        self.pictureFrame.tkraise()
        
    def show_picture_frame(self):
        self.pictureFrame.tkraise()
        
    def show_statistics_frame(self):
        self.statisticsFrame.tkraise()
        
if __name__ == "__main__":
    app = App()
    app.mainloop()