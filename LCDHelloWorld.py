from RPLCD import i2c
from time import sleep

lcdmode = 'i2c'
cols = 20
rows = 4
charmap = 'A00'
i2c_expander = 'PCF8574'

address = 0x27
port = 1

lcd = i2c.CharLCD(i2c_expander, address, port=port, charmap=charmap, cols=cols, rows=rows)

lcd.write_string('Hello World')
lcd.crlf()
lcd.write_string('IoT with Toad')
lcd.crlf()
lcd.write_string('Phppot')
sleep(5)

lcd.backlight_enabled = False
lcd.close(clear=True)
