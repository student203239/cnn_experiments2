import scipy
from matplotlib import pyplot

DataRange = range(0, 360)
DataRange = map(scipy.deg2rad, DataRange)
Data1 = map(scipy.sin, DataRange)
Data2 = map(scipy.cos, DataRange)

toggle = True

def onclick(event):
    print 'click'
    global toggle

    toggle = not toggle
    event.canvas.figure.clear()

    if toggle:
        event.canvas.figure.gca().plot(Data1)
    else:
        event.canvas.figure.gca().plot(Data2)

    event.canvas.draw()

def onkey(evt):
    print 'key'
    # import pdb
    # pdb.set_trace()

fig = pyplot.figure()
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)

pyplot.plot(Data1)
pyplot.show()