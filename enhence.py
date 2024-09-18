import os

import PIL
from matplotlib import pyplot as plt

#image_path 输入路径
#savename 保存路径

image = PIL.Image.open(image_path).convert("RGB")

f1, axes1 = plt.subplots(1, constrained_layout=True)
axes1.axis('off')

axes1.imshow(image.transpose(1, 2, 0))
f1.set_size_inches(3, 3)
f1.savefig(os.path.join(savename.split(".")[0] + "_pre" + ".png"), bbox_inches='tight', pad_inches=0)
plt.close()