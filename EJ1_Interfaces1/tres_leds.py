import Adafruit_BBIO.GPIO as GPIO
import time

GPIO.setup("USR0", GPIO.OUT)
GPIO.setup("USR1", GPIO.OUT)
GPIO.setup("USR2", GPIO.OUT)

while True:

    GPIO.output("USR0", GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output("USR0", GPIO.LOW)
    time.sleep(0.5)

    GPIO.output("USR1", GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output("USR1", GPIO.LOW)
    time.sleep(0.5)

    GPIO.output("USR2", GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output("USR2", GPIO.LOW)
    time.sleep(0.5)
