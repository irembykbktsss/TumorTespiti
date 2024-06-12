import numpy as np
import cv2 
from PIL import Image
from matplotlib import pyplot as plt
from numpy import save
from numpy import load
#from reader import reader

class HOG():  
    
    def __init__(self):

        self.patch_boyut = 64

        self.hasta = int(input("Toplamda 5 hastamız vardır. İstediğiniz hasta sayısını giriniz (0-4):"))

        if self.hasta > 4:
            print("Yanlış rakam girildi. (0-4)")
        
        if self.hasta == 0 :
            self.tumorVoksel = np.load('Tümör Tespiti/hasta1_tumor_{}.npy'.format(self.patch_boyut))
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

    def extract_features(self, extractor, voxel_data):
        max_feature_size = 0
        features = []
        for voxel in voxel_data:
            patch_image = voxel * 255
            patch_image = np.array(patch_image[16:48, 16:48], dtype=np.uint8)
            keypoints, descriptors = extractor.detectAndCompute(patch_image, None)
            if descriptors is not None:
                feature_size = descriptors.shape[0]
                if feature_size > max_feature_size:
                    max_feature_size = feature_size
                features.append(descriptors)
            else:
                features.append(np.zeros((0, 128)))
        
        # Pad features to make them all the same size
        padded_features = []
        for feature in features:
            padding = max_feature_size - feature.shape[0]
            padded_feature = np.pad(feature, ((0, padding), (0, 0)), mode='constant')
            padded_features.append(padded_feature)
        
        return np.array(padded_features)




    def tumorHog(self):       
        self.patch_boyut = 32 
        self.featureTumorList = np.zeros((self.tumorVoksel.shape[0], 324), dtype=float)
              
        for patch in range(len(self.tumorVoksel)):            
            patch_image = self.tumorVoksel[patch,:,:]
            patch_image = patch_image * 255
            patch_image = np.array(patch_image[16:48, 16:48], dtype=np.uint8)

            hog = cv2.HOGDescriptor((self.patch_boyut, self.patch_boyut), (16, 16), (8, 8), (8, 8), 9)
            featureTumor = hog.compute(patch_image)
            self.featureTumorList[patch,:] = featureTumor.transpose()
        
        plt.imshow(self.tumorVoksel[0], 'gray')                                 
        plt.show()
        plt.plot(self.featureTumorList[0])
        plt.show()

    def arkaplanHog(self):
        self.patch_boyut = 32
        self.featureArkaList = np.zeros((self.arkaplanVoksel.shape[0], 324), dtype=float) 
       
        for patch in range(len(self.arkaplanVoksel)):            
            patch_image = self.arkaplanVoksel[patch,:,:]
            patch_image = patch_image * 255
            patch_image = np.array(patch_image[16:48, 16:48], dtype=np.uint8)

            hog = cv2.HOGDescriptor((self.patch_boyut, self.patch_boyut), (16, 16), (8, 8), (8, 8), 9)
            featureArka = hog.compute(patch_image)
            self.featureArkaList[patch,:] = featureArka.transpose()
           
        plt.imshow(self.arkaplanVoksel[0], 'gray')                                 
        plt.show()  
        plt.plot(self.featureArkaList[0])
        plt.show()

    def tumorSIFT(self):
        sift = cv2.SIFT_create()
        self.featureTumorList = self.extract_features(sift, self.tumorVoksel)
        plt.imshow(self.tumorVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureTumorList[0])
        plt.show()

    def arkaplanSIFT(self):
        sift = cv2.SIFT_create()
        self.featureArkaList = self.extract_features(sift, self.arkaplanVoksel)
        plt.imshow(self.arkaplanVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureArkaList[0])
        plt.show()

    def tumorBRISK(self):
        brisk = cv2.BRISK_create()
        self.featureTumorList = self.extract_features(brisk, self.tumorVoksel)
        plt.imshow(self.tumorVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureTumorList[0])
        plt.show()

    def arkaplanBRISK(self):
        brisk = cv2.BRISK_create()
        self.featureArkaList = self.extract_features(brisk, self.arkaplanVoksel)
        plt.imshow(self.arkaplanVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureArkaList[0])
        plt.show()

    def tumorFREAK(self):
        brisk = cv2.BRISK_create()
        freak = cv2.xfeatures2d.FREAK_create()
        self.featureTumorList = []
        for voxel in self.tumorVoksel:
            patch_image = voxel * 255
            patch_image = np.array(patch_image[16:48, 16:48], dtype=np.uint8)
            keypoints = brisk.detect(patch_image, None)
            keypoints, descriptors = freak.compute(patch_image, keypoints)
            if descriptors is not None:
                self.featureTumorList.append(descriptors.flatten())
            else:
                self.featureTumorList.append(np.zeros(128))
        self.featureTumorList = np.array(self.featureTumorList)
        plt.imshow(self.arkaplanVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureTumorList[0])
        plt.show()
    
    def arkaplanFREAK(self):
        brisk = cv2.BRISK_create()
        freak = cv2.xfeatures2d.FREAK_create()
        self.featureArkaList = []
        for voxel in self.arkaplanVoksel:
            patch_image = voxel * 255
            patch_image = np.array(patch_image[16:48, 16:48], dtype=np.uint8)
            keypoints = brisk.detect(patch_image, None)
            keypoints, descriptors = freak.compute(patch_image, keypoints)
            if descriptors is not None:
                self.featureArkaList.append(descriptors.flatten())
            else:
                self.featureArkaList.append(np.zeros(128))
        self.featureArkaList = np.array(self.featureArkaList)
        plt.imshow(self.arkaplanVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureArkaList[0])
        plt.show()

    def tumorSURF(self):
        surf = cv2.xfeatures2d.SURF_create()
        self.featureTumorList = self.extract_features(surf, self.tumorVoksel)
        plt.imshow(self.tumorVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureTumorList[0])
        plt.show()

    def arkaplanSURF(self):
        surf = cv2.xfeatures2d.SURF_create()
        self.featureArkaList = self.extract_features(surf, self.arkaplanVoksel)
        plt.imshow(self.arkaplanVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureArkaList[0])
        plt.show()

    def tumorORB(self):
        orb = cv2.ORB_create()
        self.featureTumorList = self.extract_features(orb, self.tumorVoksel)
        plt.imshow(self.tumorVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureTumorList[0])
        plt.show()

    def arkaplanORB(self):
        orb = cv2.ORB_create()
        self.featureArkaList = self.extract_features(orb, self.arkaplanVoksel)
        plt.imshow(self.arkaplanVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureArkaList[0])
        plt.show()

    def tumorKAZE(self):
        kaze = cv2.KAZE_create()
        self.featureTumorList = self.extract_features(kaze, self.tumorVoksel)
        plt.imshow(self.tumorVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureTumorList[0])
        plt.show()

    def arkaplanKAZE(self):
        kaze = cv2.KAZE_create()
        self.featureArkaList = self.extract_features(kaze, self.arkaplanVoksel)
        plt.imshow(self.arkaplanVoksel[0], 'gray')
        plt.show()
        plt.plot(self.featureArkaList[0])
        plt.show()

    def file(self):
        self.patch_boyut = 32
        
        if self.hasta == 0:
            np.save('Tümör Tespiti/hasta1_featureTumor_{}.npy'.format(self.patch_boyut), self.featureTumorList, 'ab+')
            self.loadedFeatureTumor1 = np.load('Tümör Tespiti/hasta1_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('Tümör Tespiti/hasta1_featureArkaplan_{}.npy'.format(self.patch_boyut), self.featureArkaList, 'ab+')
            self.loadedFeatureArka1 = np.load('Tümör Tespiti/hasta1_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 1:
            np.save('Tümör Tespiti/hasta2_featureTumor_{}.npy'.format(self.patch_boyut), self.featureTumorList, 'ab+')
            self.loadedFeatureTumor2 = np.load('Tümör Tespiti/hasta2_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('Tümör Tespiti/hasta2_featureArkaplan_{}.npy'.format(self.patch_boyut), self.featureArkaList, 'ab+')
            self.loadedFeatureArka2 = np.load('Tümör Tespiti/hasta2_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 2:
            np.save('Tümör Tespiti/hasta3_featureTumor_{}.npy'.format(self.patch_boyut), self.featureTumorList, 'ab+')
            self.loadedFeatureTumor3 = np.load('Tümör Tespiti/hasta3_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('Tümör Tespiti/hasta3_featureArkaplan_{}.npy'.format(self.patch_boyut), self.featureArkaList, 'ab+')
            self.loadedFeatureArka3 = np.load('Tümör Tespiti/hasta3_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 3:
            np.save('Tümör Tespiti/hasta4_featureTumor_{}.npy'.format(self.patch_boyut), self.featureTumorList, 'ab+')
            self.loadedFeatureTumor4 = np.load('Tümör Tespiti/hasta4_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('Tümör Tespiti/hasta4_featureArkaplan_{}.npy'.format(self.patch_boyut), self.featureArkaList, 'ab+')
            self.loadedFeatureArka4 = np.load('Tümör Tespiti/hasta4_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 4:
            np.save('Tümör Tespiti/hasta5_featureTumor_{}.npy'.format(self.patch_boyut), self.featureTumorList, 'ab+')
            self.loadedFeatureTumor5 = np.load('Tümör Tespiti/hasta5_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('Tümör Tespiti/hasta5_featureArkaplan_{}.npy'.format(self.patch_boyut), self.featureArkaList, 'ab+')
            self.loadedFeatureArka5 = np.load('Tümör Tespiti/hasta5_featureArkaplan_{}.npy'.format(self.patch_boyut)) 

       
if __name__ == '__main__':
        Hasta1 = HOG()

        # Hasta1.tumorHog()
        # Hasta1.arkaplanHog()

        # Hasta1.tumorSIFT()
        # Hasta1.arkaplanSIFT()

        # Hasta1.tumorBRISK()   
        # Hasta1.arkaplanBRISK()

        Hasta1.tumorFREAK()
        Hasta1.arkaplanFREAK()

        # Hasta1.tumorSURF()
        # Hasta1.arkaplanSURF()

        # Hasta1.tumorORB()        OLDU
        # Hasta1.arkaplanORB()

        # Hasta1.tumorKAZE()
        # Hasta1.arkaplanKAZE()


        
               

