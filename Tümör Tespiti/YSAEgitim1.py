import numpy as np
import cv2 
from PIL import Image
from matplotlib import pyplot as plt
from numpy import save
from numpy import load

class YSAEgitim1():              #HASTA1 İÇİN EĞİTİM 

    def __init__(self):
        self.patch_boyut = 32
        
        self.loadedFeatureTumor2 = np.load('Tümör Tespiti/hasta2_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka2 = np.load('Tümör Tespiti/hasta2_featureArkaplan_{}.npy'.format(self.patch_boyut))

        self.loadedFeatureTumor3 = np.load('Tümör Tespiti/hasta3_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka3 = np.load('Tümör Tespiti/hasta3_featureArkaplan_{}.npy'.format(self.patch_boyut))

        self.loadedFeatureTumor4 = np.load('Tümör Tespiti/hasta4_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka4 = np.load('Tümör Tespiti/hasta4_featureArkaplan_{}.npy'.format(self.patch_boyut))

        self.loadedFeatureTumor5 = np.load('Tümör Tespiti/hasta5_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka5 = np.load('Tümör Tespiti/hasta5_featureArkaplan_{}.npy'.format(self.patch_boyut))


    def tumor(self):
       
        self.tumorVokselSayısı1 = np.concatenate((self.loadedFeatureTumor2,self.loadedFeatureTumor3), axis=0)
        self.tumorVokselSayısı2 = np.concatenate((self.loadedFeatureTumor4,self.loadedFeatureTumor5), axis=0)
        self.tumor = np.concatenate((self.tumorVokselSayısı1, self.tumorVokselSayısı2),axis=0)
        print(self.tumor.shape)                                               #(4079,324)  toplam tümör voksel sayısı, hog özellik sayısı    
           
    def arkaplan(self):

        self.arkaplanVokselSayısı1 = np.concatenate((self.loadedFeatureArka2,self.loadedFeatureArka3), axis=0)
        self.arkaplanVokselSayısı2 = np.concatenate((self.loadedFeatureArka4,self.loadedFeatureArka5), axis=0)
        self.arkaplan = np.concatenate((self.arkaplanVokselSayısı1, self.arkaplanVokselSayısı2),axis=0)
        print(self.arkaplan.shape)                                            #(4000,324)   toplam arkaplan voksel sayısı, hog özellik sayısı

    def egitimData(self):

        self.egitimData = np.concatenate((self.arkaplan,self.tumor), axis=0)    
        print(self.egitimData.shape)                                         #(8079,324)    toplam voksel sayıları, hog özellik sayısı

    def tumorEtiketVektor(self):             #01

        self.zerosTumor = np.zeros((self.tumor.shape[0],1), dtype=float)
        self.onesTumor = np.ones((self.tumor.shape[0],1), dtype=float)
        
        self.tumorEtiketVektor = np.concatenate((self.zerosTumor, self.onesTumor) , axis=1)
        print(self.tumorEtiketVektor.shape)                                 #(4079,2)      tumor vokselsayıları,2
        
    def arkaplanEtiketVektor(self):          #10
        
        self.zerosArka = np.zeros((self.arkaplan.shape[0],1), dtype=float)
        self.onesArka = np.ones((self.arkaplan.shape[0],1), dtype=float)

        self.arkaplanEtiketVektor = np.concatenate((self.onesArka, self.zerosArka), axis=1)
        print(self.arkaplanEtiketVektor.shape)                              #(4000,2)      arkaplan voksel sayıları,2
        
    def etiketVektor(self):
        self.etiketVektor = np.concatenate((self.arkaplanEtiketVektor , self.tumorEtiketVektor) , axis=0)
        print(self.etiketVektor.shape)                                      #(8079,2)      toplam voksel sayılar,1

    
    def createYSA(self):                                                             #boş model oluşturur

        self.YSA = cv2.ml.ANN_MLP_create()
    
    def train(self):        
                                     
        self.layer_sizes = np.int64([324, 50 , 2])                                 #özellik sayısı(girdi) , gizli katman noran sayısı, çıktı
        self.YSA.setLayerSizes(self.layer_sizes)                                     #Giriş ve çıkış katmanları dahil olmak üzere her katmanda nöron sayısını belirten tamsayı vektörü
        
        self.YSA.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM ,0,0)                   #Etkinleştirme fonksiyonu
        
        self.YSA.setTermCriteria((cv2.TermCriteria_MAX_ITER+cv2.TermCriteria_EPS, 100, 0.1))                #EPS hata değeri 0.01 altına düştüğünde dur 
           
        self.etiketVektor = np.array(self.etiketVektor, dtype=np.float32)
        self.egitimData = np.array(self.egitimData, dtype=np.float32)
        
        self.YSA.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.0001)                     #Eğitim yöntemini ve ortak parametreleri ayarlar
        self.YSA.train(self.egitimData , cv2.ml.ROW_SAMPLE , self.etiketVektor)      #istatiksel modeli eğitir , ROW_SAMPLE örneklerin satırda yer aldığı anlamında , bir satırdaki 1x324 float değer 1 örnektir anlamında.
        self.test = self.YSA.save('YSA1')
   

if __name__ == "__main__":
    Hasta = YSAEgitim1()
    Hasta.tumor()
    Hasta.arkaplan()
    Hasta.egitimData()
    Hasta.tumorEtiketVektor()
    Hasta.arkaplanEtiketVektor()
    Hasta.etiketVektor()
    Hasta.createYSA()
    Hasta.train()

