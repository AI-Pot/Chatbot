# libraries
import RPi.GPIO as GPIO
import time
from RPi_I2C_LCD_driver import RPi_I2C_driver
import Adafruit_DHT as dht

# important variables
GPIO.setmode(GPIO.BCM)
DIGIT=14
GPIO.setup(DIGIT,GPIO.IN)
lcd = RPi_I2C_driver.lcd(0x27)

try:
    while True:
        h, t = dht.read_retry(dht.DHT22,12)
        print("Humidity: {}, Temperature: {}".format(h, t))
        digit_val=GPIO.input(DIGIT)
        print("Digit Value: %d" %(digit_val))
        temp = ''
        if digit_val == 0:
            temp = 'GOOD'
        else:
            temp = 'Bad '
        lcd.lcd_display_string(" Hum  Temp  Mois ", 1)    
        lcd.lcd_display_string("{0:0.1f}% {1:0.1f}C {2}".format(h, t, temp), 2)
        time.sleep(5)
        lcd.clear()
finally:
    GPIO.cleanup()
