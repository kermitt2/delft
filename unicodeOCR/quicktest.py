from uocr import UOCR
import os

uocr = UOCR(img_w=128)
uocr.loadweights('weights/weights.h5')
fdir = os.path.dirname(__file__)
image = os.path.join(fdir, 'training_data/e10.png')
ans = uocr.ocr_frompic(image)
print(ans)