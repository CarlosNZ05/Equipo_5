import Adafruit_BBIO.GPIO as GPIO
import time

TX_PIN = "P8_10"        
BAUDRATE = 9600         
BIT_TIME = 1.0 / BAUDRATE 

def setup_uart():
    GPIO.setup(TX_PIN, GPIO.OUT)
    GPIO.output(TX_PIN, GPIO.HIGH)
    time.sleep(0.1) 

def uart_tx(byte):
    #Bit de start
    GPIO.output(TX_PIN, GPIO.LOW)
    time.sleep(BIT_TIME)

    #Bits de datos
    for i in range(8):
        bit = (byte >> i) & 1 
        
        if bit == 1:
            GPIO.output(TX_PIN, GPIO.HIGH)
        else:
            GPIO.output(TX_PIN, GPIO.LOW)
            
        time.sleep(BIT_TIME)

    #bit de stop
    GPIO.output(TX_PIN, GPIO.HIGH)
    time.sleep(BIT_TIME)


    setup_uart()
    uart_tx(ord('H'))
    time.sleep(0.05)
    uart_tx(ord('i'))
