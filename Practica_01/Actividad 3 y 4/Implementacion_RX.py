import Adafruit_BBIO.GPIO as GPIO
import time

RX_PIN = "P8_12"
BAUDRATE = 9600         
BIT_TIME = 1.0 / BAUDRATE

#estados
ESTADO_IDLE = 0
ESTADO_START = 1
ESTADO_DATOS = 2
ESTADO_STOP = 3

def setup_uart_rx():
    GPIO.setup(RX_PIN, GPIO.IN)

def uart_rx_fsm():
    estado_actual = ESTADO_IDLE
    byte_recibido = 0
    indice_bit = 0
    
    while True:
        if estado_actual == ESTADO_IDLE:
            if GPIO.input(RX_PIN) == GPIO.LOW:
                estado_actual = ESTADO_START
                
        elif estado_actual == ESTADO_START:

               time.sleep(BIT_TIME*1.5)
               indice_bit = 0
               byte_recibido = 0
               estado_actual = ESTADO_DATOS
            
        elif estado_actual == ESTADO_DATOS:
            bit = GPIO.input(RX_PIN)
            byte_recibido |= (bit << indice_bit)
            indice_bit += 1

            if indice_bit == 8:#ver si funciona sino poner un for
                estado_actual = ESTADO_STOP
            else:
                time.sleep(BIT_TIME)
                
        elif estado_actual == ESTADO_STOP:
            time.sleep(BIT_TIME * 1.5)
            return byte_recibido

setup_uart_rx()
print("Esperando...")
while True:
    byte = uart_rx_fsm()
    print("Recibido:", chr(byte), byte)
      
