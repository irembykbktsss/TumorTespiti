import numpy as np
import cv2 
from PIL import Image
from matplotlib import pyplot as plt
from numpy import save
from numpy import load
#from reader import reader

class HOG():  
    
    def __init__(self):

        self.patch_boyut = 32

        self.hasta = int(input("Toplamda 5 hastamız vardır.İstediğiniz hasta sayısını giriniz(0-4):"))

        if self.hasta > 4:
            print("Yanlış rakam girildi.(0-4)")
        
        if self.hasta == 0 :
            self.tumorVoksel = np.load('Tümör Tespiti/hasta1_tumor_{}.npy '.format(self.patch_boyut))
            self.arkaplanVoksel = np.load('Tümör Tespiti/hasta1_arkaplan_{}.npy'.format(self.patch_boyut))
        elif self.hasta == 1 :
            self.tumorVoksel = np.load('Tümör Tespiti/hasta2_tumor_{}.npy'.format(self.patch_boyut))
            self.arkaplanVoksel = np.load('Tümör Tespiti/hasta2_arkaplan_{}.npy'.format(self.patch_boyut))
        elif self.hasta == 2 :
            self.tumorVoksel = np.load('Tümör Tespiti/hasta3_tumor_{}.npy'.format(self.patch_boyut))
            self.arkaplanVoksel = np.load('Tümör Tespiti/hasta3_arkaplan_{}.npy'.format(self.patch_boyut))
        elif self.hasta == 3 :
            self.tumorVoksel = np.load('Tümör Tespiti/hasta4_tumor_{}.npy'.format(self.patch_boyut))
            self.arkaplanVoksel = np.load('Tümör Tespiti/hasta4_arkaplan_{}.npy'.format(self.patch_boyut))
        elif self.hasta == 4 :
            self.tumorVoksel = np.load('Tümör Tespiti/hasta5_tumor_{}.npy'.format(self.patch_boyut))
            self.arkaplanVoksel = np.load('Tümör Tespiti/hasta5_arkaplan_{}.npy'.format(self.patch_boyut))

    def tumorHog(self):       
        
        self.patch_boyut = 32 

        self.featureTumorList = np.zeros((self.tumorVoksel.shape[0],324) , dtype=float)
              
        for self.patch in range(len(self.tumorVoksel)):
            
            self.patch_image = self.tumorVoksel[self.patch,:,:]
            self.patch_image = self.patch_image*255                                                          #0 ila 255 arasına çekmek için
            self.patch_image = np.array(self.patch_image[16:48,16:48] , dtype=np.uint8)

            self.hog = cv2.HOGDescriptor((self.patch_boyut,self.patch_boyut),(16,16),(8,8),(8,8),9)          #varsayılan parametrelerle hog tanımlayıcı ve dedektör oluşturur
            
            self.featureTumor = self.hog.compute(self.patch_image)
            self.featureTumorList[self.patch,:] = self.featureTumor.transpose()
        print(len(self.featureTumor))
        print(self.featureTumorList)
        print(len(self.featureTumorList))          
        #print(self.featureTumorList.shape)       #(1013,1764)
        #print(self.featureTumor.shape)           #(1764,1)(satır,sütun)       

        plt.imshow(self.patch_image , 'gray')                                 
        plt.show()

        plt.plot(self.featureTumor)
        plt.show()

    def arkaplanHog(self):
        
        self.patch_boyut = 32
    
        self.featureArkaList = np.zeros((self.arkaplanVoksel.shape[0],324) , dtype=float) 
       
        for self.patch in range(len(self.arkaplanVoksel)):
            
            self.patch_image = self.arkaplanVoksel[self.patch,:,:]
            self.patch_image = self.patch_image*255                           
            self.patch_image = np.array(self.patch_image[16:48,16:48] , dtype=np.uint8)

            self.hog = cv2.HOGDescriptor((self.patch_boyut,self.patch_boyut),(16,16),(8,8),(8,8),9)       
            self.featureArka = self.hog.compute(self.patch_image)
            self.featureArkaList[self.patch,:] = self.featureArka.transpose()
           
        print(self.featureArkaList)
        print(len(self.featureArkaList)) 
        
        plt.imshow(self.patch_image , 'gray')                                 
        plt.show()  

        plt.plot(self.featureArka)
        plt.show() 

    def file (self):
        self.patch_boyut = 32
        
        if self.hasta == 0 :
            np.save('Tümör Tespiti/hasta1_featureTumor_{}.npy'.format(self.patch_boyut) , self.featureTumorList , 'ab+')
            self.loadedFeatureTumor1 = np.load('Tümör Tespiti/hasta1_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('Tümör Tespiti/hasta1_featureArkaplan_{}.npy'.format(self.patch_boyut) , self.featureArkaList , 'ab+')
            self.loadedFeatureArka1 = np.load('Tümör Tespiti/hasta1_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 1 :

            np.save('Tümör Tespiti/hasta2_featureTumor_{}.npy'.format(self.patch_boyut) , self.featureTumorList , 'ab+')
            self.loadedFeatureTumor2 = np.load('Tümör Tespiti/hasta2_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('Tümör Tespiti/hasta2_featureArkaplan_{}.npy'.format(self.patch_boyut) , self.featureArkaList , 'ab+')
            self.loadedFeatureArka2 = np.load('Tümör Tespiti/hasta2_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 2 :
            np.save('Tümör Tespiti/hasta3_featureTumor_{}.npy'.format(self.patch_boyut) , self.featureTumorList , 'ab+')
            self.loadedFeatureTumor3 = np.load('Tümör Tespiti/hasta3_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('Tümör Tespiti/hasta3_featureArkaplan_{}.npy'.format(self.patch_boyut) , self.featureArkaList , 'ab+')
            self.loadedFeatureArka3 = np.load('Tümör Tespiti/hasta3_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 3:
            np.save('Tümör Tespiti/hasta4_featureTumor_{}.npy'.format(self.patch_boyut) , self.featureTumorList , 'ab+')
            self.loadedFeatureTumor4 = np.load('Tümör Tespiti/hasta4_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('Tümör Tespiti/hasta4_featureArkaplan_{}.npy'.format(self.patch_boyut) , self.featureArkaList , 'ab+')
            self.loadedFeatureArka4 = np.load('Tümör Tespiti/hasta4_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 4:
            np.save('Tümör Tespiti/hasta5_featureTumor_{}.npy'.format(self.patch_boyut) , self.featureTumorList , 'ab+')
            self.loadedFeatureTumor5 = np.load('Tümör Tespiti/hasta5_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('Tümör Tespiti/hasta5_featureArkaplan_{}.npy'.format(self.patch_boyut) , self.featureArkaList , 'ab+')
            self.loadedFeatureArka5 = np.load('Tümör Tespiti/hasta5_featureArkaplan_{}.npy'.format(self.patch_boyut))                
    
   
if __name__ == '__main__':
    
    Hasta1 = HOG()
    
    Hasta1.tumorHog()
    Hasta1.arkaplanHog()
    Hasta1.file()   
    #  Her hasta için 1 kere çalıştır


    
   
