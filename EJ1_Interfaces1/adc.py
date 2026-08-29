import Adafruit_BBIO.ADC as ADC
import time

ADC.setup()

while True:

    valor = ADC.read("P9_39")

    print(valor)

    time.sleep(0.5)
