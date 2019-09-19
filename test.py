import numpy as np
from keras import models
from keras.models import load_model
from skimage.io import imsave
import os
import cv2
from keras import backend as K

pred_dir = 'Predicted'

def test(imgs_test):
    img_rows = 352
    img_cols = 352
    band = 2
    
    autoencoder = load_model('model-checkpoint.hdf5')

    #start prediction
    imgs_mask_test = autoencoder.predict(imgs_test, verbose=1)
    imgs_mask_test = np.array(imgs_mask_test).reshape(10,img_rows,img_cols,2)    

    #save prediction
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    idc=np.arange(0,len(imgs_mask_test))
    for image,image_id  in zip(imgs_mask_test,idc):   
        image = np.argmax(np.array(image),axis=2)
        imsave(os.path.join(pred_dir,str(image_id)+'_pred.png'), image)
        

if __name__ == '__main__':
    #path to folder containing test images
    test_path = 'CamVid/test-1/*.png'
    dir = 'TestImages'
    testdata = []
    filelist5 = glob.glob(test_path)
    save = 0
    for fname in filelist5:
        img = cv2.imread(fname)
        img = cv2.resize(img,(352,352))
        testdata.append(img)
        cv2.imwrite(os.path.join(dir,str(save)+'.png'), img)
        save = save + 1

    test_data = np.array(testdata)
    #passing input as argument
    test(test_data)
