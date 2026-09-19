import Adafruit_BBIO.GPIO as GPIO
import time

TX_PIN = "P8_10"
BAUDRATE =9600
BIT_TIME = 1.0 / BAUDRATE


def setup_uart():
    GPIO.setup(TX_PIN, GPIO.OUT)

    # UART en reposo = HIGH
    GPIO.output(TX_PIN, GPIO.HIGH)

    time.sleep(0.1)


def uart_tx(byte):
    # Bit de START
    GPIO.output(TX_PIN, GPIO.LOW)
    time.sleep(BIT_TIME)

    # 8 bits de datos, LSB primero
    for i in range(8):
        bit = (byte >> i) & 1

        if bit == 1:
            GPIO.output(TX_PIN, GPIO.HIGH)
        else:
            GPIO.output(TX_PIN, GPIO.LOW)

        time.sleep(BIT_TIME)

    # Bit de STOP
    GPIO.output(TX_PIN, GPIO.HIGH)
    time.sleep(BIT_TIME*2)


setup_uart()
while True:
    uart_tx(ord('I'))
    time.sleep(0.05)
