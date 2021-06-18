# soil moisture
import RPi.GPIO as GPIO
import time
import spidev

GPIO.setmode(GPIO.BCM)
DIGIT=14
GPIO.setup(DIGIT,GPIO.IN)
#spi=spidev.SpiDev()
#spi.open(0,0)
#spi.max_speed_hz=50000

#def read_spi_adc(adcChannel):
    #adcValue=0
    #buff=spi.xfer2([1,(8+adcChannel)<<4,0])
    #adcValue=((buff[1]&3)<<8)+buff[2]
    #return adcValue

try:
    while True:
        #adcValue=read_spi_adc(0)
        #print("soil: %d" %(adcValue))
        digit_val=GPIO.input(DIGIT)
        print("Digit Value: %d" %(digit_val))
        time.sleep(0.5)
        
finally:
    GPIO.cleanup()
    spi.close()