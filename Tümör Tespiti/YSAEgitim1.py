import numpy as np
import cv2 
from PIL import Image
from matplotlib import pyplot as plt
from numpy import save
from numpy import load
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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
        self.layer_sizes = np.int64([324, 50 , 2])                                 #özellik sayısı(girdi) , gizli katman noran sayısı, çıktı
        self.YSA.setLayerSizes(self.layer_sizes)                                     #Giriş ve çıkış katmanları dahil olmak üzere her katmanda nöron sayısını belirten tamsayı vektörü
        
        self.YSA.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM ,0,0)                   #Etkinleştirme fonksiyonu
        
        self.YSA.setTermCriteria((cv2.TermCriteria_MAX_ITER+cv2.TermCriteria_EPS, 100, 0.1))                #EPS hata değeri 0.01 altına düştüğünde dur 
           
        self.etiketVektor = np.array(self.etiketVektor, dtype=np.float32)
        self.egitimData = np.array(self.egitimData, dtype=np.float32)
        
        self.YSA.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.0001)                     #Eğitim yöntemini ve ortak parametreleri ayarlar
        self.YSA.train(self.egitimData , cv2.ml.ROW_SAMPLE , self.etiketVektor)      #istatiksel modeli eğitir , ROW_SAMPLE örneklerin satırda yer aldığı anlamında , bir satırdaki 1x324 float değer 1 örnektir anlamında.
        self.test = self.YSA.save('YSA1')
    
    def train_and_evaluate(self):
        # Modeli ve verileri ayarlayın
        self.createYSA()  # Model oluşturma
        
        # 10 kat çapraz doğrulama
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        accuracies = []
        
        # Verilerin veri tipini np.float32 olarak dönüştürün
        self.egitimData = np.array(self.egitimData, dtype=np.float32)
        self.etiketVektor = np.array(self.etiketVektor, dtype=np.float32)
        
        for train_index, test_index in kf.split(self.egitimData):
            # Eğitim ve test verilerini ayırma
            X_train, X_test = self.egitimData[train_index], self.egitimData[test_index]
            y_train, y_test = self.etiketVektor[train_index], self.etiketVektor[test_index]
            
            # Modeli eğitme
            self.YSA.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
            
            # Modeli test etme
            _, y_pred = self.YSA.predict(X_test)
            
            # Test sonuçlarını değerlendirme
            acc = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            accuracies.append(acc)
        
        # Ortalama doğruluk hesaplama
        mean_accuracy = np.mean(accuracies)
        print(f"10-fold cross-validation mean accuracy: {mean_accuracy * 100:.2f}%")


if __name__ == "__main__":
    Hasta = YSAEgitim1()
    Hasta.tumor()
    Hasta.arkaplan()
    Hasta.egitimData()
    Hasta.tumorEtiketVektor()
    Hasta.arkaplanEtiketVektor()
    Hasta.etiketVektor()
    Hasta.createYSA()
    Hasta.train_and_evaluate()

