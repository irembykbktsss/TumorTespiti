import pydicom
import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np
import random

class TumorDetector:
    def __init__(self, dicom_paths, mha_paths, patch_size=64):
        self.dicom_paths = dicom_paths
        self.mha_paths = mha_paths
        self.patch_size = patch_size

    def read_dicom(self, path):
        reader = sitk.ImageSeriesReader()
        filenamesDICOM = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(filenamesDICOM)
        dicom_image = reader.Execute()
        dicom_array = sitk.GetArrayFromImage(dicom_image)
        return dicom_array.astype(float)

    def read_mha(self, path):
        mha_image = sitk.ReadImage(path)
        mha_array = sitk.GetArrayFromImage(mha_image)
        return mha_array.astype(float)

    def normalize_image(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = (image - min_val) / (max_val - min_val)
        return normalized_image

    def plot_images(self, image, rows=6, cols=6):
        for i in range(len(image)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image[i], cmap='gray')
            plt.xticks([]), plt.yticks([])
        plt.show()

    def cumulative_histogram(self, image):
        flattened = image.flatten()
        hist, bins = np.histogram(flattened, bins=256)
        cumsum = np.cumsum(hist)
        cumsum_normalized = (cumsum - cumsum.min()) / (cumsum.max() - cumsum.min())
        return cumsum_normalized, bins

    def get_threshold(self, cumsum_normalized, bins):
        for i in range(len(cumsum_normalized)):
            if cumsum_normalized[i] >= 0.90:
                threshold = bins[i + 1]
                break
        return threshold

    def apply_threshold(self, image, threshold):
        image[image > threshold] = threshold
        return image

    def extract_patches(self, norm_image, label_image, tumor_voxel_count, background_voxel_count):
        tumor_patches = np.zeros((3000, self.patch_size, self.patch_size), dtype=float)
        background_patches = np.zeros((3000, self.patch_size, self.patch_size), dtype=float)
        tumor_count = 0
        background_count = 0
        
        depth, height, width = norm_image.shape
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    if label_image[i, j, k] == 1 and tumor_count < 3000:
                        if (self.patch_size / 2) <= j <= (height - self.patch_size / 2) and (self.patch_size / 2) <= k <= (width - self.patch_size / 2):
                            rand_tumor = random.randrange(0, int(tumor_voxel_count / 1000))
                            if rand_tumor == 10:
                                tumor_patches[tumor_count] = norm_image[i, j - int(self.patch_size / 2):j + int(self.patch_size / 2), k - int(self.patch_size / 2):k + int(self.patch_size / 2)]
                                tumor_count += 1
                    elif label_image[i, j, k] == 0 and background_count < 3000:
                        if (self.patch_size / 2) <= j <= (height - self.patch_size / 2) and (self.patch_size / 2) <= k <= (width - self.patch_size / 2):
                            rand_background = random.randrange(0, int(background_voxel_count / 1000))
                            if rand_background == 100:
                                background_patches[background_count] = norm_image[i, j - int(self.patch_size / 2):j + int(self.patch_size / 2), k - int(self.patch_size / 2):k + int(self.patch_size / 2)]
                                background_count += 1
        
        tumor_patches = tumor_patches[:tumor_count]
        background_patches = background_patches[:background_count]
        
        return tumor_patches, background_patches

    def save_patches(self, tumor_patches, background_patches, patient_index):
        np.save(f'TumorPatches_Patient{patient_index + 1}_{self.patch_size}.npy', tumor_patches)
        np.save(f'BackgroundPatches_Patient{patient_index + 1}_{self.patch_size}.npy', background_patches)


    def process_patient(self, patient_index):
        dicom_image = self.read_dicom(self.dicom_paths[patient_index])
        mha_image = self.read_mha(self.mha_paths[patient_index])

        cumsum_normalized, bins = self.cumulative_histogram(dicom_image)
        threshold = self.get_threshold(cumsum_normalized, bins)
        dicom_image = self.apply_threshold(dicom_image, threshold)
        normalized_image = self.normalize_image(dicom_image)

        tumor_voxel_counts = [27541, 32466, 33284, 22161, 22867]
        background_voxel_counts = [820331, 852270, 814588, 788847, 825005]

        tumor_patches, background_patches = self.extract_patches(normalized_image, mha_image, tumor_voxel_counts[patient_index], background_voxel_counts[patient_index])
        self.save_patches(tumor_patches, background_patches, patient_index)

    def process_all_patients(self):
        for i in range(len(self.dicom_paths)):
            print(f"Processing patient {i + 1}")
            self.process_patient(i)


# Specify the paths to DICOM and MHA files
dicom_paths = [
    'C:/Users/MONSTER/Desktop/GRANDTRUTH/TCGA-02-0011-1.HASTA - GRANDTRUTH YAPILDI/02-01-1998-MRI BRAIN WWO CONTRAMR-31709/3-GRANDTRUTH YAPILDI-AX T2 FSE-77488/',
    'C:/Users/MONSTER/Desktop/GRANDTRUTH/TCGA-02-0034-2.HASTA - GRANDTRUTH YAPILDI/07-27-1997-MRI BRAIN WWO CONTRAMR-39956/10-GRANDTRUTH YAPILDI-AX T2 FSE-01030/',
    'C:/Users/MONSTER/Desktop/GRANDTRUTH/TCGA-02-0047-3.HASTA - GRANDTRUTH YAPILDI/12-15-1998-MRI BRAIN WWO CONTR-70492/3-GRANDTRUTH YAPILDI-AX T2 FSE-79920/',
    'C:/Users/MONSTER/Desktop/GRANDTRUTH/TCGA-02-0116-4.HASTA - GRANDTRUTH YAPILDI/03-22-1997-MRI BRAIN WWO CONTRAMR-70515/3-GRANDTRUTH YAPILDI-AX T2 FSE-42524/',
    'C:/Users/MONSTER/Desktop/GRANDTRUTH/TCGA-06-0139-6.HASTA - GRANDTRUTH YAPILDI/03-24-2005-19928-GRANDTURTH YAPILDI/5-GRANDTURTH YAPILDI AX T2 FR-FSE RF2 150-15143/'
]

mha_paths = [
    'C:/Users/MONSTER/Desktop/GRANDTRUTH/TCGA-02-0011-1.HASTA - GRANDTRUTH YAPILDI/02-01-1998-MRI BRAIN WWO CONTRAMR-31709/3-GRANDTRUTH YAPILDI-AX T2 FSE-77488/Hasta1.mha',
    'C:/Users/MONSTER/Desktop/GRANDTRUTH/TCGA-02-0034-2.HASTA - GRANDTRUTH YAPILDI/07-27-1997-MRI BRAIN WWO CONTRAMR-39956/10-GRANDTRUTH YAPILDI-AX T2 FSE-01030/Hasta2.mha',
    'C:/Users/MONSTER/Desktop/GRANDTRUTH/TCGA-02-0047-3.HASTA - GRANDTRUTH YAPILDI/12-15-1998-MRI BRAIN WWO CONTR-70492/3-GRANDTRUTH YAPILDI-AX T2 FSE-79920/Hasta3.mha',
    'C:/Users/MONSTER/Desktop/GRANDTRUTH/TCGA-02-0116-4.HASTA - GRANDTRUTH YAPILDI/03-22-1997-MRI BRAIN WWO CONTRAMR-70515/3-GRANDTRUTH YAPILDI-AX T2 FSE-42524/Hasta4.mha',
    'C:/Users/MONSTER/Desktop/GRANDTRUTH/TCGA-06-0139-6.HASTA - GRANDTRUTH YAPILDI/03-24-2005-19928-GRANDTURTH YAPILDI/5-GRANDTURTH YAPILDI AX T2 FR-FSE RF2 150-15143/Hasta6.mha'
]


# Örnek oluşturma
tumor_detector = TumorDetector(dicom_paths, mha_paths, patch_size=64)

# Tüm hastaları işlemek için yöntemi çağırma
tumor_detector.process_all_patients()
