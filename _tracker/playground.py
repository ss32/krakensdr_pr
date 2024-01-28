import numpy as np
import pickle
import matplotlib.pylab as plt
from matplotlib import cm
import matplotlib.colors as colors
from skimage.transform import resize
from PIL import Image
fin =  open('/tmp/range_maps/1706221945.5_CAFMAtrix.pkl','rb')
data = pickle.load(fin)
fin.close()


youssef_color_map = ['#000020', '#000030', '#000050', '#000091', '#1E90FF', '#FFFFFF', '#FFFF00', '#FE6D16', '#FE6D16', '#FF0000',
                     '#FF0000', '#C60000', '#9F0000', '#750000', '#4A0000']

color_map = colors.ListedColormap(youssef_color_map)
scalarMap = cm.ScalarMappable(cmap=color_map)
CAFMatrixLog = resize(data,(1280,1280),order=1, anti_aliasing=True) 
seg_colors = scalarMap.to_rgba(CAFMatrixLog) 
img = Image.fromarray(np.uint8(seg_colors*255))
img.show()