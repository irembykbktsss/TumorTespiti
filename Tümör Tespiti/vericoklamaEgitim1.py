import numpy as np
import cv2 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

class YSAEgitim1():
    def __init__(self):
        self.patch_boyut = 32
        
        # Tümör ve arkaplan özelliklerini yükle
        self.loadedFeatureTumor2 = np.load(f'TumorPatches_Patient2_{self.patch_boyut}.npy')
        self.loadedFeatureTumor3 = np.load(f'TumorPatches_Patient3_{self.patch_boyut}.npy')
        self.loadedFeatureTumor4 = np.load(f'TumorPatches_Patient4_{self.patch_boyut}.npy')
        self.loadedFeatureTumor5 = np.load(f'TumorPatches_Patient5_{self.patch_boyut}.npy')

        self.loadedFeatureArka2 = np.load(f'BackgroundPatches_Patient2_{self.patch_boyut}.npy')
        self.loadedFeatureArka3 = np.load(f'BackgroundPatches_Patient3_{self.patch_boyut}.npy')
        self.loadedFeatureArka4 = np.load(f'BackgroundPatches_Patient4_{self.patch_boyut}.npy')
        self.loadedFeatureArka5 = np.load(f'BackgroundPatches_Patient5_{self.patch_boyut}.npy')

    def tumor(self):
        # Tüm tümör verilerini birleştir
        self.tumor = np.concatenate((self.loadedFeatureTumor2, self.loadedFeatureTumor3,
                                     self.loadedFeatureTumor4, self.loadedFeatureTumor5), axis=0)
        print("Tumor shape:", self.tumor.shape)  # Tümör veri şekli

    def arkaplan(self):
        # Tüm arkaplan verilerini birleştir
        self.arkaplan = np.concatenate((self.loadedFeatureArka2, self.loadedFeatureArka3,
                                        self.loadedFeatureArka4, self.loadedFeatureArka5), axis=0)
        print("Arkaplan shape:", self.arkaplan.shape)  # Arkaplan veri şekli

    def egitimData(self):
        # Eğitim verisini tüm tümör ve arkaplan verilerini birleştirerek oluştur
        self.egitimData = np.concatenate((self.tumor, self.arkaplan), axis=0)
        print("Eğitim verisi shape:", self.egitimData.shape)  # Eğitim veri şekli

    def tumorEtiketVektor(self):
        # Tüm tümör verileri için etiket vektörünü oluştur (1,0)
        self.tumorEtiketVektor = np.hstack((np.zeros((self.tumor.shape[0], 1), dtype=np.float32),
                                            np.ones((self.tumor.shape[0], 1), dtype=np.float32)))
        print("Tumor etiket vektörü shape:", self.tumorEtiketVektor.shape)  # Tümör etiket vektörü şekli

    def arkaplanEtiketVektor(self):
        # Tüm arkaplan verileri için etiket vektörünü oluştur (0,1)
        self.arkaplanEtiketVektor = np.hstack((np.ones((self.arkaplan.shape[0], 1), dtype=np.float32),
                                               np.zeros((self.arkaplan.shape[0], 1), dtype=np.float32)))
        print("Arkaplan etiket vektörü shape:", self.arkaplanEtiketVektor.shape)  # Arkaplan etiket vektörü şekli

    def etiketVektor(self):
        # Tüm etiket vektörlerini birleştir
        self.etiketVektor = np.concatenate((self.tumorEtiketVektor, self.arkaplanEtiketVektor), axis=0)
        print("Toplam etiket vektörü shape:", self.etiketVektor.shape)  # Toplam etiket vektörü şekli

    def createYSA(self):
        # Veri türlerini ve formatlarını kontrol etmek için ekstra print işlemleri
        print("Eğitim Veri Türü (Başlangıçta):", self.egitimData.dtype)
        print("Eğitim Veri Şekli (Başlangıçta):", self.egitimData.shape)

        print("Etiket Vektörü Türü (Başlangıçta):", self.etiketVektor.dtype)
        print("Etiket Vektörü Şekli (Başlangıçta):", self.etiketVektor.shape)

        # Veri türlerini np.float32 olarak dönüştürme
        self.egitimData = np.array(self.egitimData, dtype=np.float32)

        # Veriyi 2D formatına dönüştürme (10381, 32, 32) -> (10381, 1024)
        self.egitimData = self.egitimData.reshape(self.egitimData.shape[0], -1)
        
        # Veri türlerini kontrol etme
        print("Eğitim Veri Türü (Dönüştürüldükten sonra):", self.egitimData.dtype)
        print("Eğitim Veri Şekli (Dönüştürüldükten sonra):", self.egitimData.shape)

        # YSA modelini oluşturma ve eğitme adımları devam ediyor...
        self.YSA = cv2.ml.ANN_MLP_create()
        self.layer_sizes = np.int64([self.egitimData.shape[1], 50, 2])  # Giriş boyutunu otomatik olarak ayarlayın
        self.YSA.setLayerSizes(self.layer_sizes)
        self.YSA.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0, 0)
        self.YSA.setTermCriteria((cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 100, 0.1))
        self.YSA.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.0001)
        
        # Eğitim verilerini ve etiket vektörlerini YSA modeline verme
        self.YSA.train(self.egitimData, cv2.ml.ROW_SAMPLE, self.etiketVektor)

        # Modeli kaydetme
        self.test = self.YSA.save('YSA1-v2')


    def train_and_evaluate(self):
        # Modeli ve verileri eğit ve değerlendir
        self.createYSA()

        # 10 kat çapraz doğrulama
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in kf.split(self.egitimData):
            X_train, X_test = self.egitimData[train_index], self.egitimData[test_index]
            y_train, y_test = self.etiketVektor[train_index], self.etiketVektor[test_index]

            # Modeli eğit
            self.YSA.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

            # Modeli test et ve doğruluğu hesapla
            _, y_pred = self.YSA.predict(X_test)
            acc = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            accuracies.append(acc)

        # Ortalama doğruluğu yazdır
        mean_accuracy = np.mean(accuracies)
        print(f"10 kat çapraz doğrulama ortalama doğruluk: {mean_accuracy * 100:.2f}%")

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
