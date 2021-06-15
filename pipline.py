import RPi.GPIO as GPIO
import time
import os

btnPin = 18

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(btnPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def press_button():
    GPIO.wait_for_edge(btnPin,GPIO.RISING,bouncetime=100)
    time.sleep(0.1)
    
    if GPIO.input(btnPin) == 0:
        os.system('python3 tts.py')

while True:
    press_button()
    