from uocr import UOCR
import datetime

uocr = UOCR()
run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
start_epoch = 0
stop_epoch = 20
uocr.train(run_name, start_epoch, stop_epoch)