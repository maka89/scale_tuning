import numpy as np
import matplotlib.pyplot as plt

just = {
    1:16/15,
    2:9/8,
    3:6/5,
    4:5/4,
    5:4/3,
    7:3/2,
    9:5/3,
    11:15/8,
    12:2
}
names = {1:"Minor Second",2:"Major Second", 3:"Minor Third",4:"Major Third", 5:"Forth", 7:"Fifth", 9:"Major Sixth",11:"Major Seventh",12:"Octave"}
equal = { i: np.abs(2**(i/12)) for i in range(0,13)}

xmin = 0.98
xmax = 1.02

x=np.linspace(xmin,xmax,101)
h = np.abs((2*x**(-5.0) )**(1/12))
w = x*h*h

plots = {1:h,2:w,3:w*h,4:w*w,5:w*w*h,7:w*w*h*w,9:w*w*h*w*w,11:w*w*h*w*w*w,12:w*w*h*w*w*w*h}


def ratio_to_cents(x):
    return 1200*np.log2(x)

f,ax = plt.subplots(len(names.keys()),1)
keys = list(names.keys())
for i in range(0,len(keys)):
    just_cents = ratio_to_cents(just[keys[i]])
    equal_cents = ratio_to_cents(equal[keys[i]])
    plot_cents = ratio_to_cents(plots[keys[i]])

    ax[i].plot(x,plot_cents-just_cents)
    ax[i].hlines(equal_cents-just_cents,xmin,xmax,colors="b",linestyles="dotted")
    ax[i].hlines(0,xmin,xmax,colors="g",linestyles="dotted")
    
    
    ax[i].vlines(1.0,-5,5,colors="r",linestyles="dotted")
    ax[i].set_title(names[keys[i]])
    if i != len(keys)-1:
        ax[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax[0].legend([ "New T.","Equal T.","Just T."])

plt.show()