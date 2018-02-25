import initial
import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
#import tensorflow as tf

i=1

if i==1:
    json_file = open('model_arch.json','r')
    loaded_model_json = json_file.read()
#    json_file.close()
    loaded_model = model_from_json(loaded_model_json,custom_objects={'ZeroPadding': initial.ZeroPadding,'CorrnetCost': initial.CorrnetCost})
    loaded_model.load_weights("model_wts.h5")
    print("Loaded Model from disk")
    #mfp = open('/home/shwegarg/data_files/model_arch.json','r')
    #mj = mfp.read()
    #model2 = model_from_json(mj, {'ZSumLayer': ZSumLayer})
else:
    from keras.models import load_model
    loaded_model=load_model('corrnet_model1.h5',custom_objects={'ZeroPadding': initial.ZeroPadding,'CorrnetCost': initial.CorrnetCost})
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
                



left_view,_=initial.prepare_data()
bre
initial.reconstruct_from_left(loaded_model,left_view[6:7])
# reconstruct_from_right(model,right_view[6:7])
