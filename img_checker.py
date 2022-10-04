import imutils
import numpy as np
import cv2
import tkinter
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
from imutils import paths
from PIL import Image, ImageTk
import os
from functools import partial
import traceback

rootWindow=tk.Tk()
rootWindow.title("IMAGE CONTROLLER")

class MenuFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master=master
        
        self.rgbimg=self.dst=self.filepath=None
        self.loadedimgNum=self.nowimgNum=self.previmgNum=0
        
        self.canny_mode=self.binary_mode=self.contrast_mode=self.sharpness_mode=self.detail_mode=self.blur_mode=self.hat_mode=False
        
        self.changeValue=''
        self.changeValue1=np.array([1, 255])
        self.changeValue2=np.array([100])
        self.changeValue3=np.array([0, 1])
        
        self.createWidgets()

    def toCanny(self):
        try:
            self.binary_mode=self.contrast_mode=self.sharpness_mode=self.detail_mode=self.blur_mode=self.hat_mode=False           
            if not self.canny_mode:
                self.changeValue=''
                self.changeValue1=np.array([1, 255])
                self.changeValue2=np.array([100])
                self.changeValue3=np.array([0, 1])                  
                self.changeValue=self.changeValue1
                self.createWidgets2()
                self.canny_mode=True 
            else:           
                if self.rgbimg is not None:
                    gray=cv2.cvtColor(self.rgbimg, cv2.COLOR_BGR2GRAY)
                    blurred=cv2.GaussianBlur(gray, (5, 5), 0)
                    self.dst=cv2.Canny(blurred, self.changeValue[0], self.changeValue[1], apertureSize=3) 
                    
                    try:
                        minLineLength=100
                        maxLineGap=20
                        lines=cv2.HoughLinesP(self.dst,1,np.pi/180,130,minLineLength,maxLineGap)
                        for line in lines:
                            x1,y1,x2,y2=line[0]
                            cv2.line(self.dst,(x1,y1),(x2,y2),(0,255,0),2)
                    except:
                        pass

                    cv_img=cv2.cvtColor(self.dst, cv2.COLOR_GRAY2RGB)
                    self.showImg(cv_img)
                    # print(f'to Canny: {self.changeValue[0], self.changeValue[1]}')
        except:
            print(traceback.format_exc())
            pass

    def toBinary(self):
        try:
            self.canny_mode=self.contrast_mode=self.sharpness_mode=self.detail_mode=self.blur_mode=self.hat_mode=False           
            if not self.binary_mode:
                self.changeValue=''
                self.changeValue1=np.array([1, 255])
                self.changeValue2=np.array([100])
                self.changeValue3=np.array([0, 1])                 
                self.changeValue=self.changeValue1
                self.createWidgets2()
                self.binary_mode=True
            else:           
                if self.rgbimg is not None:        
                    gray=cv2.cvtColor(self.rgbimg, cv2.COLOR_BGR2GRAY)
                    _, self.dst=cv2.threshold(gray, self.changeValue[0], self.changeValue[1], cv2.THRESH_BINARY)

                    cv_img=cv2.cvtColor(self.dst, cv2.COLOR_GRAY2RGB)
                    self.showImg(cv_img)
                    # print(f'to Binary: {self.changeValue[0], self.changeValue[1]}')
        except:
            print(traceback.format_exc())
            pass

    def toContrast(self):
        try:
            self.canny_mode=self.binary_mode=self.sharpness_mode=self.detail_mode=self.blur_mode=self.hat_mode=False       
            if not self.contrast_mode:
                self.changeValue=''
                self.changeValue1=np.array([1, 255])
                self.changeValue2=np.array([100])
                self.changeValue3=np.array([0, 1])              
                self.changeValue=self.changeValue2
                self.createWidgets2()
                self.contrast_mode=True
            else:           
                if self.rgbimg is not None:                
                    lab=cv2.cvtColor(self.rgbimg, cv2.COLOR_BGR2LAB) 
                    l, a, b=cv2.split(lab) 
                    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(self.changeValue[0], self.changeValue[0])) 
                    cl=clahe.apply(l) 
                    self.dst=cv2.merge((cl, a, b)) 

                    cv_img=cv2.cvtColor(self.dst, cv2.COLOR_LAB2BGR) 
                    self.showImg(cv_img)
                    # print(f'to Contrast: {self.changeValue[0], self.changeValue[1]}')
        except:
            print(traceback.format_exc())
            pass

    def toSharpness(self):
        try:
            self.canny_mode=self.binary_mode=self.contrast_mode=self.detail_mode=self.blur_mode=self.hat_mode=False          
            if not self.sharpness_mode:
                self.changeValue=''
                self.changeValue1=np.array([1, 255])
                self.changeValue2=np.array([100])
                self.changeValue3=np.array([0, 1]) 
                self.changeValue=self.changeValue3
                self.createWidgets2()        
                self.sharpness_mode=True
            else:           
                if self.rgbimg is not None:   
                    sharpening_mask=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    self.dst=cv2.filter2D(self.rgbimg, -1, sharpening_mask)  

                    cv_img=cv2.cvtColor(self.dst, cv2.COLOR_RGB2BGR)
                    self.showImg(cv_img)
                    # print(f'to Sharpness')
        except:
            print(traceback.format_exc())
            pass

    def toDetail(self):
        try: 
            self.canny_mode=self.binary_mode=self.contrast_mode=self.sharpness_mode=self.blur_mode=self.hat_mode=False             
            if not self.detail_mode:
                self.changeValue=''
                self.changeValue1=np.array([1, 255])
                self.changeValue2=np.array([100])
                self.changeValue3=np.array([0, 1])              
                self.changeValue=self.changeValue1
                self.createWidgets2()
                self.detail_mode=True
            else:           
                if self.rgbimg is not None:               
                    self.dst=cv2.detailEnhance(self.rgbimg, self.changeValue[0], self.changeValue[1]) # /255

                    cv_img=cv2.cvtColor(self.dst, cv2.COLOR_RGB2BGR)
                    self.showImg(cv_img)
                    # print(f'to Detail: {self.changeValue[0], self.changeValue[1]}')
        except:
            print(traceback.format_exc())
            pass

    def toBlur(self):
        try: 
            self.canny_mode=self.binary_mode=self.contrast_mode=self.sharpness_mode=self.detail_mode=self.hat_mode=False             
            if not self.blur_mode:
                self.changeValue=''
                self.changeValue1=np.array([1, 255])
                self.changeValue2=np.array([100])
                self.changeValue3=np.array([0, 1])                  
                self.changeValue=self.changeValue3
                self.createWidgets2()               
                self.blur_mode=True
            else:           
                if self.rgbimg is not None: 
                    self.dst=cv2.medianBlur(self.rgbimg, 1)

                    cv_img=cv2.cvtColor(self.dst, cv2.COLOR_RGB2BGR)
                    self.showImg(cv_img)            
                    # print(f'to Detail')
        except:
            print(traceback.format_exc())
            pass

    def toHat(self):
        try: 
            self.canny_mode=self.binary_mode=self.contrast_mode=self.sharpness_mode=self.detail_mode=self.blur_mode=False         
            if not self.hat_mode:
                self.changeValue=''
                self.changeValue1=np.array([1, 255])
                self.changeValue2=np.array([100])
                self.changeValue3=np.array([0, 1])                 
                self.changeValue=self.changeValue2
                self.createWidgets2()
                self.hat_mode=True
            else:           
                if self.rgbimg is not None: 
                    k=cv2.getStructuringElement(cv2.MORPH_RECT, (self.changeValue[0], self.changeValue[0]))
                    self.dst=cv2.morphologyEx(self.rgbimg, cv2.MORPH_BLACKHAT, k)    

                    cv_img=cv2.cvtColor(self.dst, cv2.COLOR_RGB2BGR)
                    self.showImg(cv_img)
                    # print(f'to Hat: {self.changeValue[0], self.changeValue[1]}')    
        except:
            print(traceback.format_exc())
            pass                        

    def showImg(self, cv_img):
        self.photoimg=ImageTk.PhotoImage(image=Image.fromarray(cv_img))
        self.mainWorkcanvas.itemconfig(self.backgroundImage, image=self.photoimg)        

    def loadImages(self, filepath=None):
        try:
            if filepath is not None:
                filename=filepath.split(os.path.sep)[-1]
                self.rgbimg=cv2.imread(filepath, cv2.IMREAD_COLOR)
                self.rgbimg=imutils.resize(self.rgbimg, width=1024)
                print(f"[INFO] current file name: {filename}")
            else:
                self.canny_mode=self.binary_mode=self.contrast_mode=self.sharpness_mode=self.detail_mode=self.blur_mode=self.hat_mode=False
                
                self.changeValue=''
                self.changeValue1=np.array([1, 255])
                self.changeValue2=np.array([100])
                self.changeValue3=np.array([0, 1])                
                try:
                    self.thresholdFrame.destroy()
                    for n, inputValue in enumerate(self.changeValue):
                        self.valueScale[n].destroy()
                except:
                    pass

            (H, W)=self.rgbimg.shape[:2]
            self.mainWorkcanvas.configure(width=int(W), height=int(H))

            cv_img=cv2.cvtColor(self.rgbimg, cv2.COLOR_BGR2RGB)
            self.showImg(cv_img)

        except:
            print(traceback.format_exc())
            pass

    def onselectListbox(self, event):
        w=event.widget
        try:
            self.previmgNum=self.nowimgNum
            self.nowimgNum=w.curselection()[0]
            self.filepath=w.get(self.nowimgNum)
            self.loadImages(self.filepath)
            self.listboxStateshow.configure(text='{} / {} Images Loaded'.format(self.nowimgNum+1, self.loadedimgNum))
        except:
            print(traceback.format_exc())
            pass

    def scrollValue(self, inputValue, n):
        self.changeValue[n]=float(inputValue)
        if self.canny_mode: self.toCanny()
        if self.binary_mode: self.toBinary()
        if self.contrast_mode: self.toContrast()
        if self.sharpness_mode: self.toSharpness()
        if self.detail_mode: self.toDetail()
        if self.blur_mode: self.toBlur()
        if self.hat_mode: self.toHat()

    def saveImages(self, is_folder=False):
        try:
            if self.filepath is not None:
                filepath = self.filepath
                # print('1', filepath)
                folderpath=filepath.split(os.path.sep)[-2]
                foldername=filepath.split(sep='/')[-2]
                # print('2', folderpath)

                if self.canny_mode: new_foldrname = 'canny_img'
                elif self.binary_mode: new_foldrname = 'binary_img'
                elif self.contrast_mode: new_foldrname = 'contrast_img' 
                elif self.sharpness_mode: new_foldrname = 'sharpness_img'
                elif self.detail_mode: new_foldrname = 'detail_img'
                elif self.blur_mode: new_foldrname = 'blur_img' 
                elif self.hat_mode: new_foldrname = 'hat_img'
                else: new_foldrname = 'ori_img'

                writePath = f'save_img/{foldername}_{new_foldrname}'
                os.makedirs(writePath, exist_ok=True)
                if is_folder:
                    img_paths = sorted(list(paths.list_images(folderpath)))
                    for img_path in img_paths:
                        # image = cv2.imread(img_path)
                        self.rgbimg=cv2.imread(img_path, cv2.IMREAD_COLOR)
                        self.rgbimg=imutils.resize(self.rgbimg, width=1024)
                        filename=img_path.split(os.path.sep)[-1]

                        if self.canny_mode: self.toCanny()
                        if self.binary_mode: self.toBinary()
                        if self.contrast_mode: self.toContrast()
                        if self.sharpness_mode: self.toSharpness()
                        if self.detail_mode: self.toDetail()
                        if self.blur_mode: self.toBlur()
                        if self.hat_mode: self.toHat()
                        
                        print(f"[INFO] saving file: {filename}")
                        cv2.imwrite(f'{writePath}/{filename}', self.dst)
                    print()
                    print(f"[INFO] saving completed/ now value: {self.changeValue}")
                    print()
                else:
                    # image = cv2.imread(filepath)
                    filename=filepath.split(os.path.sep)[-1]
                    cv2.imwrite(f'{writePath}/{filename}', self.dst)
        except:
            print(traceback.format_exc())
            pass

    def dirforder(self):
        try:
            fordername=filedialog.askdirectory()
            self.imagePaths=sorted(list(paths.list_images(fordername)))
            
            if self.imagePaths:
                for (num, imagePath) in enumerate(self.imagePaths):
                    self.filelistbox.insert(num, imagePath)

                self.loadedimgNum=len(self.imagePaths)
                self.cropPositiondata=[[] for i in range(self.loadedimgNum)]
                self.cropNamedata=[[] for i in range(self.loadedimgNum)]
                self.listboxStateshow.configure(text='{} / {} Images Loaded'.format(self.nowimgNum+1, self.loadedimgNum))

                self.filelistbox.select_set(0)
                self.filelistbox.event_generate("<<ListboxSelect>>")
                # self.imageSelectBTN.configure(state='disabled')
        except:
            print(traceback.format_exc())
            pass

    def createWidgets2(self):
        start_val=1
        end_val=self.changeValue[-1]
        self.valueScale=[]
        try:
            self.thresholdFrame.destroy()
            for n, inputValue in enumerate(self.changeValue):
                self.valueScale[n].destroy()
        except:
            pass

        # choose threshold values
        self.thresholdFrame=tk.LabelFrame(self.mainFrame, text='Threshold 조정')
        self.thresholdFrame.pack(fill='both', expand='no', padx=5, pady=5)

        for n, inputValue in enumerate(self.changeValue):
            self.valueScale.append(n)
            self.valueScale[n]=tkinter.Scale(self.thresholdFrame, command=lambda inputValue, n=n: self.scrollValue(inputValue, n), orient="horizontal", showvalue=True, from_=start_val, to=end_val, resolution=1)
            self.valueScale[n].set(inputValue)
            self.valueScale[n].pack(side='top', padx=5, pady=10, fill='x')

    def createWidgets(self):
        self.mainWorkcanvas=tk.Canvas(self.master, width=0, height=0, cursor="cross")
        self.backgroundImage=self.mainWorkcanvas.create_image(0, 0, image='', anchor='nw')
        self.mainWorkcanvas.grid(row=0, column=1, sticky='nw')

        self.mainFrame=tk.Frame(self.master)
        self.mainFrame.grid(row=0, column=0, sticky='nw')

        # choose options
        self.chooseOptionFrame=tk.LabelFrame(self.mainFrame, text='이미지 전처리 방법 선택')
        self.chooseOptionFrame.pack(fill='both', expand='no', padx=5, pady=5)
        self.cannyBTN=tk.Radiobutton(self.chooseOptionFrame, text='CANNY', value=0, command=self.toCanny, indicatoron=False, width=18)
        self.binaryBTN=tk.Radiobutton(self.chooseOptionFrame, text='BINARY', value=1, command=self.toBinary, indicatoron=False, width=18)
        self.contrastBTN=tk.Radiobutton(self.chooseOptionFrame, text='CANTRAST', value=2, command=self.toContrast, indicatoron=False, width=18)
        self.sharpnessBTN=tk.Radiobutton(self.chooseOptionFrame, text='SHARPNESS', value=3, command=self.toSharpness, indicatoron=False, width=18, state='disable')
        self.detailBTN=tk.Radiobutton(self.chooseOptionFrame, text='DETAIL', value=4, command=self.toDetail, indicatoron=False, width=18, state='disable')
        self.blurBTN=tk.Radiobutton(self.chooseOptionFrame, text='BLUR', value=5, command=self.toBlur, indicatoron=False, width=18, state='disable')
        self.hatBTN=tk.Radiobutton(self.chooseOptionFrame, text='HAT', value=6, command=self.toHat, indicatoron=False, width=18, state='disable')
        self.oriBTN=tk.Radiobutton(self.chooseOptionFrame, text='ORIGINAL', command=self.loadImages, indicatoron=False, width=18)
        
        self.cannyBTN.grid(row=0,column=0, padx=5, pady=5)
        self.binaryBTN.grid(row=0,column=1, padx=5, pady=5)
        self.contrastBTN.grid(row=1,column=0, padx=5, pady=5)
        self.sharpnessBTN.grid(row=1,column=1, padx=5, pady=5)
        self.detailBTN.grid(row=2,column=0, padx=5, pady=5)
        self.blurBTN.grid(row=2,column=1, padx=5, pady=5)
        self.hatBTN.grid(row=3,column=0, padx=5, pady=5)
        self.oriBTN.grid(row=3,column=1, padx=5, pady=5)

        # load image files
        self.loadImageFrame=tk.LabelFrame(self.mainFrame, text='이미지 불러오기')
        self.loadImageFrame.pack(fill='both', expand='no', padx=5, pady=5)
        self.imageSelectBTN=tk.Button(self.loadImageFrame, text='이미지 폴더 선택', command=self.dirforder, takefocus=False, width=18)
        self.imageSelectBTN.pack(side='left', fill='both', padx=5, pady=5)

        # save image files
        self.saveImageFrame=tk.LabelFrame(self.mainFrame, text='이미지 저장하기')
        self.saveImageFrame.pack(fill='both', expand='no', padx=5, pady=5)
        self.imageSaveBTN=tk.Button(self.saveImageFrame, text='현재 이미지 저장', command=self.saveImages, takefocus=False, width=18)
        self.folderSaveBTN=tk.Button(self.saveImageFrame, text='폴더 이미지 저장', command=partial(self.saveImages, True), takefocus=False, width=18)
        self.imageSaveBTN.pack(side='left', fill='both', padx=5, pady=5)
        self.folderSaveBTN.pack(side='right', fill='both', padx=5, pady=5)

        # load image list
        self.filelistLabelframe=tk.LabelFrame(self.mainFrame, text='이미지 파일 목록')
        self.filelistLabelframe.pack(fill='both', expand='no', padx=5, pady=5)
        self.listboxStateshow=tk.Label(self.filelistLabelframe, text='No Images Loaded')
        self.listboxStateshow.pack(side='top', padx=2, pady=2)
        self.filelistboxscrollbarH=tk.Scrollbar(self.filelistLabelframe, orient='vertical')
        self.filelistboxscrollbarH.pack(side="right", fill="y")
        self.filelistboxscrollbarW=tk.Scrollbar(self.filelistLabelframe, orient='horizontal')
        self.filelistboxscrollbarW.pack(side="bottom", fill="x")
        self.filelistbox=tk.Listbox(self.filelistLabelframe, selectmode='single', height=15,
                                        yscrollcommand=self.filelistboxscrollbarH.set, xscrollcommand=self.filelistboxscrollbarW.set, takefocus=False)
        self.filelistbox.pack(fill='both', padx=5, pady=5)
        self.filelistbox.bind('<<ListboxSelect>>', self.onselectListbox)


mainWindow=MenuFrame(master=rootWindow)

rootWindow.resizable(0, 0)
rootWindow.mainloop()
