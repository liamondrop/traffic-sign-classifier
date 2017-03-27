import pickle
import cv2
import numpy as np
from sklearn.utils import shuffle

training_file = "./dataset/train.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']

assert(len(X_train) == len(y_train))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))

def rand_warp_img(img, degree):
    s = img.shape[0]
    d = lambda: np.random.randint(0, (s*degree))

    pts1 = np.float32([[0,0],[s,0],[s,s],[0,s]])
    diff = np.float32([[d(),d()],[-d(),d()],[-d(),-d()],[d(),-d()]])
    pts2 = pts1 + diff
    M = cv2.getPerspectiveTransform(pts2, pts1)
    out = cv2.warpPerspective(img, M, (32,32))
    return out

def augment_dataset(X, y):
    X_aug = np.copy(X)
    y_aug = np.copy(y)
    for i in range(X.shape[0]):
        if i % 100 == 0:
            print("TURN {}".format(i))
        label = y_aug[i]
        for j in range(3):
            aug_img = rand_warp_img(X_aug[i], .3)
            X_aug = np.append(X_aug, [aug_img], axis=0)
            y_aug = np.append(y_aug, label)
        if i % 1000 == 0:
            print("EPOCH {} ...".format(i))
    assert X_aug.shape[0] == y_aug.shape[0]
    return shuffle(X_aug, y_aug)

print(X_train.shape)
print(y_train.shape)
X_aug, y_aug = augment_dataset(X_train, y_train)
print(X_aug.shape)
print(y_aug.shape)

np.save('dataset/X_aug.npy', X_aug)
np.save('dataset/y_aug.npy', y_aug)
