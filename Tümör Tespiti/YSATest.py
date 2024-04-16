import numpy as np
import cv2 
from PIL import Image
from matplotlib import pyplot as plt
from numpy import save
from numpy import load

class YSATest():

    def __init__(self):

        self.patch_boyut = 32 

        self.loadedFeatureTumor5 = np.load('Tümör Tespiti/hasta3_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka5 = np.load('Tümör Tespiti/hasta3_featureArkaplan_{}.npy'.format(self.patch_boyut))

    
    def testData(self):
    
        self.testData = np.concatenate((self.loadedFeatureArka5,self.loadedFeatureTumor5), axis=0)    
        print(self.testData.shape)                                         #(2044,324)    toplam voksel sayıları, hog özellik sayısı 
        
    
    def tumorEtiketVektor(self):             #01
    
        self.tumorEtiketVektor = np.concatenate((np.zeros((self.loadedFeatureTumor5.shape[0],1), dtype=float), np.ones((self.loadedFeatureTumor5.shape[0],1), dtype=float)), axis=1)    
        print(self.tumorEtiketVektor.shape)                                 #(1039,2)      tumor vokselsayıları,2
        
    
    def arkaplanEtiketVektor(self):         #10

        self.arkaplanEtiketVektor = np.concatenate((np.ones((self.loadedFeatureArka5.shape[0],1), dtype=float),np.zeros((self.loadedFeatureArka5.shape[0],1), dtype=float)), axis=1)
        print(self.arkaplanEtiketVektor.shape)                              #(1005,2)      arkaplan voksel sayıları,2  
        
    
    def etiketVektor(self):
        self.testEtiket = np.concatenate((self.arkaplanEtiketVektor , self.tumorEtiketVektor) , axis=0)
        print(self.testEtiket.shape)                                      #(2044,2)      toplam voksel sayılar,1

    
    def test(self):                                                             

        self.YSA = cv2.ml.ANN_MLP_load('YSA5')      
                
        print(self.YSA.getTrainMethod())                #0
        print(self.YSA.getLayerSizes())                 #[[324][100][2]]
        print(self.YSA.getTermCriteria())               #(3, 300, 0.001)
        
        self.retval, self.result = self.YSA.predict(self.testData)           #Sağlanan örnek (ler) için yanıt (lar) ı tahmin eder
        
        print(self.result)                #sonuç matrisi        
        print(self.result.shape)          #(2044,2)


        self.result = np.round(self.result)             #yuvarlama yapıldı
                
        self.karmasaMatrisi = np.zeros([2,2], dtype=int)

        for x in range(0,len(self.testData)):
            self.pred = self.result[x]           # [1 0] gibi bir YSA tahmin dizisi
            self.truth = self.testEtiket[x]      # [1 0] gibi bir gerçek etiket dizisi

            if self.truth[0]==1 and self.truth[1]==0 and self.pred[0]==1 and self.pred[1]==0:    #[1 0] arka plan etiketiydi ve arka plan  tahmin edilmişse TN'ye sayılır
                self.karmasaMatrisi[1,1]=self.karmasaMatrisi[1,1]+1 
            if self.truth[0]==0 and self.truth[1]==1 and self.pred[0]==0 and self.pred[1]==1:   #[0 1] tümör etiketiydi ve tümör tahmin edildiyse TP'ye sayılır
                self.karmasaMatrisi[0,0]=self.karmasaMatrisi[0,0]+1 
            if self.truth[0]==1 and self.truth[1]==0 and self.pred[0]==0 and self.pred[1]==1:    #[1 0] arka plan etiketiydi ve tümör tahmin edildiyse FP'ye sayılır
                self.karmasaMatrisi[0,1]=self.karmasaMatrisi[0,1]+1 
            if self.truth[0]==0 and self.truth[1]==1 and self.pred[0]==1 and self.pred[1]==0:    #[0 1] tümör etiketiydi ve arka plan tahmin edildiyse FN'ye sayılır
                self.karmasaMatrisi[1,0]=self.karmasaMatrisi[1,0]+1 
        
        print(self.karmasaMatrisi) 
        self.kesinlik = 100*(self.karmasaMatrisi[0,0]+self.karmasaMatrisi[1,1])/np.sum(self.karmasaMatrisi,axis=None)          #sistemin yüzde başarısı
        print(self.kesinlik)         
        
        
        self.check = np.concatenate((self.result , self.testEtiket) , axis=1)
        print(self.check) 
               
        
        
if __name__ == "__main__":
    Test = YSATest()
    Test.testData()
    Test.tumorEtiketVektor()
    Test.arkaplanEtiketVektor()
    Test.etiketVektor()
    Test.test()


    