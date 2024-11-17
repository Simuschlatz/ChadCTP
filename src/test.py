import keyboard
from time import time
import matplotlib.pyplot as plt
dts = []
t1 = time()
t2 = 0
def on_key_press(e):
    global t1, t2
    t2 = time()
    dts.append(t2 - t1)
    t1 = t2
keyboard.on_press(on_key_press)
keyboard.wait('esc')
plt.scatter(range(len(dts)), dts)
plt.show()