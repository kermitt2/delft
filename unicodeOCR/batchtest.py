from uocr import UOCR

uocr = UOCR(img_w=128)
uocr.loadweights('/Users/azhar/work/delft/unicodeOCR/weights/2018:10:02:16:46:44/weights07.h5')
ans = uocr.test_batch()
print ans