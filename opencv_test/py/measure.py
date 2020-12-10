import numpy as np
import cv2
import time
from tqdm import tqdm, trange

img_amount = 360
exp_times = 100

w = 128
while w < 2049:

    h = 128
    while h < 2049:

        #print("w = " + str(w) + " / h = " + str(h))

        result = np.empty(0,int)

        #w = int(input("input width of img >>"))
        #h = int(input("input height of img >>"))

        padded = 1
        while padded < (w*2):
            padded = padded * 2

        img = np.full((h,w,img_amount),65535.0)

        for i in trange(exp_times, desc="w = " + str(w) + " / h = " + str(h)):
            start = time.perf_counter()

            #pad = np.full((h,padded - w,img_amount),0.0)
            pad = np.zeros((h,padded - w,img_amount))
            dst = np.concatenate([img, pad],1)
    
            res = time.perf_counter() - start
            result = np.append(result, res)

        print(np.average(result))
        #cv2.imwrite('test.tif', dst[:,:,0:1].astype(np.uint16))
        #print(dst[:,:,0:1])

        h = h * 2
    
    w = w * 2