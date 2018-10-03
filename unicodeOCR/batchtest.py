from uocr import UOCR

uocr = UOCR(img_w=128)
uocr.loadweights('weights/weights.h5')
ans = uocr.test_batch()
print(ans)