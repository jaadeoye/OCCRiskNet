import numpy as np
from tensorflow import keras
import imageio.v2 as sci
import PIL

#Attn weight extraction
get_alpha_layer_output = K.function([model.layers[0].input], [model.get_layer("self-attn-MIL_HE").output]) #for H and E only #to be replicated for other channels if necessary
layers_out = get_alpha_layer_output([batch[0]]) #np array for image patches

#normalization
x = np.asarray(layers_out)
minimum = x.min()
maximum = x.max()
y = (x - minimum ) / ( maximum - minimum )
y = np.round(y, 2)
z=np.asarray(batch[2]) #np array for patch coordinate
a = np.append(y,z)
print('Attn_weights for test data', a)
