import datetime
import Adafruit_DHT as dht

# DHT_SENSOR = Adafruit_DHT.DHT22
# DHT_PIN=12
wtime = datetime.datetime.now()

h, t = dht.read_retry(dht.DHT22,12)
print(wtime, 'Temp={0:0.1f}*C Humidity={0:0.1f}%'.format(t, h))


