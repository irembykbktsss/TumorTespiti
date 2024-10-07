import numpy as np
import cv2 
from PIL import Image
from matplotlib import pyplot as plt
from numpy import save
from numpy import load
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

class YSATest():

    def __init__(self):
        self.patch_boyut = 32 
        self.loadedFeatureTumor5 = np.load('Tümör Tespiti/hasta2_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka5 = np.load('Tümör Tespiti/hasta2_featureArkaplan_{}.npy'.format(self.patch_boyut))

    def testData(self):
        self.testData = np.concatenate((self.loadedFeatureArka5, self.loadedFeatureTumor5), axis=0)    
        print(self.testData.shape)  # (2044,324)

    def tumorEtiketVektor(self):  # 01
        self.tumorEtiketVektor = np.concatenate((np.zeros((self.loadedFeatureTumor5.shape[0], 1), dtype=float), np.ones((self.loadedFeatureTumor5.shape[0], 1), dtype=float)), axis=1)    
        print(self.tumorEtiketVektor.shape)  # (1039,2)

    def arkaplanEtiketVektor(self):  # 10
        self.arkaplanEtiketVektor = np.concatenate((np.ones((self.loadedFeatureArka5.shape[0], 1), dtype=float), np.zeros((self.loadedFeatureArka5.shape[0], 1), dtype=float)), axis=1)
        print(self.arkaplanEtiketVektor.shape)  # (1005,2)

    def etiketVektor(self):
        self.testEtiket = np.concatenate((self.arkaplanEtiketVektor, self.tumorEtiketVektor), axis=0)
        print(self.testEtiket.shape)  # (2044,2)

    def test(self):                                                             
        self.YSA = cv2.ml.ANN_MLP_load('YSA2xml')      
        print(self.YSA.getTrainMethod())  # 0
        print(self.YSA.getLayerSizes())  # [[324][100][2]]
        print(self.YSA.getTermCriteria())  # (3, 300, 0.001)
        
        self.retval, self.result = self.YSA.predict(self.testData)  # Tahminler
        print(self.result)  # Sonuç matrisi        
        print(self.result.shape)  # (2044,2)

        self.result = np.round(self.result)  # Yuvarlama yapıldı

        # Gerçek ve tahmin edilen etiketleri tek boyutlu hale getir
        self.truth_labels = np.argmax(self.testEtiket, axis=1)
        self.pred_labels = np.argmax(self.result, axis=1)

        self.karmasaMatrisi = np.zeros([2, 2], dtype=int)

        for x in range(len(self.testData)):
            self.pred = self.pred_labels[x]
            self.truth = self.truth_labels[x]

            if self.truth == 0 and self.pred == 0:
                self.karmasaMatrisi[0, 0] += 1  # True Negative
            if self.truth == 1 and self.pred == 1:
                self.karmasaMatrisi[1, 1] += 1  # True Positive
            if self.truth == 0 and self.pred == 1:
                self.karmasaMatrisi[0, 1] += 1  # False Positive
            if self.truth == 1 and self.pred == 0:
                self.karmasaMatrisi[1, 0] += 1  # False Negative
        
        print(self.karmasaMatrisi)

        # Hesaplamalar
        self.kesinlik = 100 * (self.karmasaMatrisi[1, 1] + self.karmasaMatrisi[0, 0]) / np.sum(self.karmasaMatrisi, axis=None)
        print(f"Accuracy: {self.kesinlik}%")

        accuracy = accuracy_score(self.truth_labels, self.pred_labels)
        sensitivity = recall_score(self.truth_labels, self.pred_labels)
        specificity = self.karmasaMatrisi[0, 0] / (self.karmasaMatrisi[0, 0] + self.karmasaMatrisi[0, 1])
        f1 = f1_score(self.truth_labels, self.pred_labels)

        print(f"Accuracy: {accuracy}")
        print(f"Sensitivity (Recall): {sensitivity}")
        print(f"Specificity: {specificity}")
        print(f"F1 Score: {f1}")

        self.check = np.concatenate((self.result, self.testEtiket), axis=1)
        print(self.check)

if __name__ == "__main__":
    Test = YSATest()
    Test.testData()
    Test.tumorEtiketVektor()
    Test.arkaplanEtiketVektor()
    Test.etiketVektor()
    Test.test()
