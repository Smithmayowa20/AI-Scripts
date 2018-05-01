import os, sys
import pickle
from PIL import Image
import cv2
import requests
from google.colab import files
import numpy as np

file_1 = "train1.txt"
file_2 = "train_answer1.txt"
file_3 = "test1.txt"
file_4 = "test_answer1.txt"

x_train_list_urls = open(file_1).read().strip().split("\n")
y_train_list_nos = open(file_2).read().strip().split("\n")

x_test_list_urls = open(file_3).read().strip().split("\n")
y_test_list_nos = open(file_4).read().strip().split("\n")

x_train_list = []
y_train_list = []

x_test_list = []
y_test_list = []


img_path = "train"
img_path_2 = "test"

  
i = 0
while i < len(x_train_urls):
  url = x_train_list_urls[i]
  answer = int(y_train_list_nos[i]) - 1
  try:
    # try to download the image
    r = requests.get(url, timeout=60)
		# save the image to disk
    p = os.path.sep.join([img_path, "{}.jpg".format(
       str(i).zfill(8))])
    f = open(p, "wb")
    f.write(r.content)
    f.close()
    size = (224, 224)
    try:
      im = Image.open(p)
      im = im.resize(size)
      im.save(p)
      img = cv2.imread(p)
      x_train_list.append(img)
      y_train_list.append(answer)
    except IOError:
      print("cannot create thumbnail for '{}'".format(p))
    f.close()
 
		# update the counter
    print("[INFO] downloaded: {}".format(p))
    i += 1
  except ConnectionError:
    print("[INFO] error downloading {}...skipping".format(p))
    print("ConnectionError")
    i += 1
  except :
    print("[INFO] error downloading {}...skipping".format(p))
    print("InvalidSchema")
    i += 1
  # handle if any exceptions are thrown during the download process

x_train_list = np.stack(x_train_list)
y_train_list = np.stack(y_train_list)

print(x_train_list.shape,"x_train_list")
print(y_train_list.shape,"y_train_list")

with open('x_train2000.pickle', 'wb') as handle:
    pickle.dump(x_train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('y_train2000.pickle', 'wb') as handle:
    pickle.dump(y_train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
  
  
j = 0
while j < len(x_test_list_urls):
  url = x_test_list_urls[j]
  answer = int(y_test_list_nos[j]) - 1
  try:
    # try to download the image
    r = requests.get(url, timeout=60)
		# save the image to disk
    p = os.path.sep.join([img_path_2, "{}.jpg".format(
       str(j).zfill(8))])
    f = open(p, "wb")
    f.write(r.content)
    f.close()
    size = 224, 224
    try:
      im = Image.open(p)
      im = im.resize(size)
      im.save(p)
      img = cv2.imread(p)
      x_test_list.append(img)
      y_test_list.append(answer)
    except IOError:
      print("cannot create thumbnail for '{}'".format(p))
    f.close()
 
		# update the counter
    print("[INFO] downloaded: {}".format(p))
    j += 1
  except ConnectionError:
    print("[INFO] error downloading {}...skipping".format(p))
    print("ConnectionError")
    j += 1
  except :
    print("[INFO] error downloading {}...skipping".format(p))
    print("InvalidSchema")
    j += 1
  # handle if any exceptions are thrown during the download process
  

x_test_list = np.stack(x_test_list)
y_test_list = np.stack(y_test_list)

print(x_test_list.shape,"x_test_list")
print(y_test_list.shape,"y_test_list")  


with open('x_test2000.pickle', 'wb') as handle:
    pickle.dump(x_test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('y_test2000.pickle', 'wb') as handle:
    pickle.dump(y_test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)