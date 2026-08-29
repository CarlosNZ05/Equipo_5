import Adafruit_BBIO.GPIO as GPIO
import time

GPIO.setup("P8_16", GPIO.IN, pull_up_down=GPIO.PUD_UP)

while True:
    estado = GPIO.input("P8_16")

    if estado == GPIO.LOW:
        print("1")
    else:
        print("0")

    time.sleep(0.5)
