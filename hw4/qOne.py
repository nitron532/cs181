import numpy as np
import cv2
from keras.datasets import cifar10
from matplotlib import pyplot as plt
import pickle
import os


def showMPL(image):
    plt.imshow(image, cmap = "gray", interpolation = "nearest")
    plt.show()
# Fixed seed for reproducibility


SEED = 42
np.random.seed(SEED)
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Ensure labels are 1D arrays
y_train = y_train.flatten()
y_test = y_test.flatten()
# We will use x_train for generating CodeWords and x_train, y_train
# for the BoF model (training histograms).
# The classification will be performed on a single test sample.
# Fixed test sample index
# We will use index 4 for x_test
TEST_SAMPLE_IDX = 4
fixed_test_image = x_test[TEST_SAMPLE_IDX]
fixed_test_label = y_test[TEST_SAMPLE_IDX]

xTrainGray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in x_train]
xTestGray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in x_test]

#1
sift = cv2.SIFT_create()
allXDescriptors = []
imgToDesc = []
descToKP = {}
finalCentroids = []
k = 16
cache_file = "trueMean.pkl"

#since we seed the random gen, we can use pickle to cache results so I dont have to wait 5 minutes every time
if os.path.exists(cache_file):
    # Load cached descriptors and keypoints
    with open(cache_file, "rb") as f:
        allXDescriptors = pickle.load(f)
        # descToKP = pickle.load(f) only need descToKP for finalCentroids, if pickle loads finalCentroids we dont need it
        finalCentroids = pickle.load(f)
        imgToDesc = pickle.load(f)

else:
    for image in xTrainGray:
        kp, des = sift.detectAndCompute(image, None)
        if des is None:
            continue

        # Limit to 100 descriptors if more
        if len(des) > 100:
            n100_idx = np.random.choice(len(des), size=100, replace=False)
        else:
            n100_idx = np.arange(len(des))

        imgToDesc.append([])
        for idx in n100_idx:
            descriptor = des[idx]  # shape (128,) → still a NumPy array
            imgToDesc[-1].append(descriptor)
            desc_tuple = tuple(descriptor.tolist())  
            descToKP[desc_tuple] = (kp[idx], image)
            allXDescriptors.append(descriptor)  # keep numeric for k-means
#2
    allXDescriptors = np.array(allXDescriptors, dtype=np.float32)
    allXDescriptors /= np.linalg.norm(allXDescriptors, axis=1, keepdims=True) + 1e-8

    descriptorSample = np.array(allXDescriptors)
    if len(descriptorSample) > 500000:
        indices = np.random.choice(len(descriptorSample), size=500000, replace=False)
        descriptorSample = descriptorSample[indices]

    startCentroidsIndices = np.random.randint(500000, size = k) #indices of randomly selected centroids in descriptorSample
    startCentroids = np.array([descriptorSample[idx] for idx in startCentroidsIndices])

    clusterLists = [[startCentroids[_]] for _ in range(k)] #List[List[128D ndarray]]

    #extract true mean
    for i in range(30): #30 iterations of kmeans
        for point in descriptorSample: #cluster points
            distances = np.linalg.norm(startCentroids - point)
            distances = np.array(distances,dtype=float)
            pointsCluster = distances.argmin()
            clusterLists[pointsCluster].append(point)

        for j in range(k): #mean centroid update
            startCentroids[j] = np.mean(clusterLists[j],axis = 0)

    finalCentroids = [[] for _ in range(k)]

    #convert true mean centroids to closest descriptor
    for i in range(k):
        distancesToMean = np.linalg.norm(clusterLists[i] - startCentroids[i], axis = 1)
        newCentroid = clusterLists[i][distancesToMean.argmin()]

    unused_descs = list(descToKP.keys())

    for i in range(k):
        if tuple(startCentroids[i].tolist()) not in descToKP:
            all_descs = np.array(unused_descs)
            distances = np.linalg.norm(all_descs - startCentroids[i], axis=1)
            nearest_idx = distances.argmin()
            nearest = tuple(all_descs[nearest_idx])
            kp = descToKP[nearest]
            # Remove so it’s not reused
            unused_descs.pop(nearest_idx)

        else:
            kp = descToKP[tuple(startCentroids[i].tolist())]

        kpData = {
            "pt": kp[0].pt,
            "size": kp[0].size,
            "angle": kp[0].angle,
            "response": kp[0].response,
            "octave": kp[0].octave,
            "class_id": kp[0].class_id
        }
        finalCentroids[i] = (startCentroids[i],kpData, kp[1]) #kp[1] is the image, kpData is the cv2keypoint data structure


    with open(cache_file, "wb") as f:
        pickle.dump(allXDescriptors, f)
        # pickle.dump(descToKP, f)
        pickle.dump(finalCentroids, f)
        pickle.dump(imgToDesc, f)


# for group in finalCentroids:
#     display = group[2][int(group[1]["pt"][1]-8):int(group[1]["pt"][1]+8), int(group[1]["pt"][0]-8):int(group[1]["pt"][0]+8)]
#     showMPL(display)

specificallyCentroids = [finalCentroids[i][0] for i in range(len(finalCentroids))] #16 detected 128d descriptor centroids 
specificallyCentroids = np.array(specificallyCentroids)
histograms = []
for descList in imgToDesc:
    bins = [0 for _ in range(k)]
    for desc in descList:
        distancesToCentroids = np.linalg.norm(specificallyCentroids - desc, axis = 1) #compare descriptor to each centroid
        bins[distancesToCentroids.argmin()] +=1  #simulate putting it in the bin by incrementing bin count
    bins = np.array(bins).astype(np.float64)
    if bins.sum() > 0:
        bins = bins / bins.sum()
    histograms.append(bins) #append this image's bins to the histogram collection
histograms = np.array(histograms)


fixed_test_image_gray = cv2.cvtColor(fixed_test_image, cv2.COLOR_BGR2GRAY)
testKP, testDes = sift.detectAndCompute(fixed_test_image_gray, None)


bins = [0 for _ in range(k)]
for desc in testDes:
    distancesToCentroids = np.linalg.norm(specificallyCentroids - desc, axis = 1)
    bins[distancesToCentroids.argmin()] += 1
bins = np.array(bins).astype(np.float64)


if bins.sum() > 0:
    bins  = bins / bins.sum()
#now bins is the histogram for this test image

distances = []
idx = 0
for hist in histograms:
    distances.append((np.linalg.norm(bins - hist, ord = 1),hist, xTrainGray[idx]))
    idx+=1

distances.sort(key = lambda x: x[0])
distances = distances[:10]

for i in range(len(distances)):
    print(distances[i][1].argmax())
    print(distances[i][0])
    print(i)
    print()
    showMPL(distances[i][2])



# featureOccurences = [0 for _ in range(k)]
# sumOfAllDetectedFeatureAppearances = 0
# for topDistances in distances:
#     kthFeature = 0
#     for featureCount in topDistances[1]:
#         featureOccurences[kthFeature]+= featureCount
#         kthFeature+=1
#         sumOfAllDetectedFeatureAppearances += featureCount

# majorityFeature = np.argmax(featureOccurences)
#majority voting, or if there isn't a single feature that appears > N/2
# majorityFeature = finalCentroids[majorityFeature]

# print(fixed_test_label)
# print(np.argmax(featureOccurences))

# #feature detected
# showMPL(fixed_test_image_gray)
# showMPL(majorityFeature[2][int(majorityFeature[1]["pt"][1]-8):int(majorityFeature[1]["pt"][1]+8), int(majorityFeature[1]["pt"][0]-8):int(majorityFeature[1]["pt"][0]+8)])

#test histogram
# plt.figure(figsize=(8, 5))
# plt.bar(np.arange(len(bins)), bins, color='steelblue', edgecolor='black')
# plt.title("Normalized BoF Histogram for Test Image (16 visual words)")
# plt.xlabel("Visual Word Index")
# plt.ylabel("Normalized Frequency")
# plt.xticks(np.arange(len(bins)))
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()



