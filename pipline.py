import RPi.GPIO as GPIO
import time
import os
import RPi_I2C_driver
import Adafruit_DHT as dht
import subprocess

all_black     = [0b11111,0b11111,0b11111,0b11111,0b11111,0b11111,0b11111,0b11111]
upper_black   = [0b11111,0b11111,0b11111,0b11111,0b00000,0b00000,0b00000,0b00000]
under_black   = [0b00000,0b00000,0b00000,0b00000,0b11111,0b11111,0b11111,0b11111]
left_to_right = [0b00000,0b00000,0b00000,0b00000,0b00001,0b00011,0b00111,0b01111]
right_to_left = [0b00000,0b00000,0b00000,0b00000,0b10000,0b11000,0b11100,0b11110]

lcd = RPi_I2C_driver.lcd(0x27)

GPIO.setmode(GPIO.BCM)
DIGIT=14 # WATER CHECK PIN
btnPin = 18 # BUTTON PIN
GPIO.setup(DIGIT,GPIO.IN)
GPIO.setwarnings(False)
GPIO.setup(btnPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)                                              

lcd.createChar(0, all_black)
lcd.createChar(1, upper_black)
lcd.createChar(2, under_black)
lcd.createChar(3, left_to_right)
lcd.createChar(4, right_to_left)
lcd.cursor()

def info_t_h_w():
    lcd.clear()
    
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
    time.sleep(2)
    lcd.clear()
    lcd.lcd_display_string("Wanna talk ", 1)    
    lcd.lcd_display_string("Push button 5s", 2)
    
    
def default_lcd_display():
    lcd.clear()
    lcd.noCursor()
    
    lcd.setCursor(3,0)
    lcd.write(0)
    
    lcd.setCursor(6,1)
    lcd.write(1)

    lcd.setCursor(7,1)
    lcd.write(2)

    lcd.setCursor(8,1)
    lcd.write(2)

    lcd.setCursor(9,1)
    lcd.write(1)

    lcd.setCursor(12,0)
    lcd.write(0)

    time.sleep(2)
    lcd.noCursor()
    


while True:
    default_lcd_display()
    info_t_h_w()

    GPIO.wait_for_edge(btnPin,GPIO.RISING,bouncetime=100)
    time.sleep(1)
    
    if GPIO.input(btnPin) == 0:
        os.system('python3 tts.py')
        #press_button()
        
GPIO.cleanup() # WHEN CUT-OFF, SHUT-OFF GPIO
  
    