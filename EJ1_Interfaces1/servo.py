import Adafruit_BBIO.PWM as PWM
import time

PWM.start("P9_14", 5, 50)

while True:

    PWM.set_duty_cycle("P9_14", 2.5)
    time.sleep(2)

    PWM.set_duty_cycle("P9_14", 7.5)
    time.sleep(2)

    PWM.set_duty_cycle("P9_14", 12.5)
    time.sleep(2)
