#import numpy as np
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
def ratio_to_cents(x):
    return 1200*np.log2(x)
def cents_to_ratio(x):
    return 2**(x/1200)

class MajorScaleGenerator:
    def __init__(self,config,octave_locked=True,order=2):
        self.config=config
        self.out = None
        self.octave_locked = octave_locked
        self.order=order
        self.offsets = {0:"C",1:"C#_Db",2:"D",3:"D#_Eb",4:"E",5:"F",6:"F#_Gb",7:"G",8:"G#_Ab",9:"A",10:"A#_Bb",11:"B"}
    def optimize(self):

        def fn(x,args):
            if self.octave_locked:
                x=np.concatenate((x[0:-1], np.array([1200-np.sum(x[0:-1])])))
            conf = args[0]
            retval = 0.0
            for c in conf:
                if c["weight"] is not None:
                    retval += np.mean( np.abs(c["weight"]*(np.dot(c["matrix"],x)-c["target"]))**self.order )
            return retval
        grad_fn = grad(fn,0)
        x0= np.array([ ratio_to_cents(x) for x in [1,2.0**(2.0/12),2.0**(4.0/12),2.0**(5.0/12),2.0**(7.0/12),2.0**(9.0/12),2.0**(11.0/12)]])
        self.res = minimize(fn,x0=x0,jac=grad_fn,args=[self.config],method="l-bfgs-b",options={"ftol":1e-32,"gtol":1e-32})
        if self.octave_locked:
            self.res["x"][-1] = 1200-np.sum(self.res["x"][0:-1])
        tmp = np.cumsum(self.res["x"])
        self.x = np.zeros(7)
        for i in range(1,7):
            self.x[i] = tmp[i-1]
        print(self.res)

    def print_report(self,disp=True):
        #print(self.res["x"])
        x = self.x
        x_comp = np.array([ ratio_to_cents(x) for x in [1,2.0**(2.0/12),2.0**(4.0/12),2.0**(5.0/12),2.0**(7.0/12),2.0**(9.0/12),2.0**(11.0/12)]])
        x_just = np.array([ratio_to_cents(x) for x in [1,9.0/8,5.0/4,4.0/3,3.0/2,5.0/3,15.0/8]])
        strrs=[]
        for c in self.config:
            vals = np.dot(c["matrix"],self.res["x"])-c["target"]
            strrs.append(c["name"])
            strrs.append("Min: {0:.02f}  Max: {1:.02f}  Avg: {2:.02f}  MAE: {3:.02f}  Equal Temp: {4:.02f}".format(np.min(vals),np.max(vals),np.mean(vals), np.mean(np.abs(vals)), 100*c["semis"]-c["target"]))
            strrs.append(" ")
        if disp:
            for st in strrs:
                print(st)
        return strrs


    def get_keys(self,offset=0):
        assert(offset<12)
        cents = self.x
        keys = {}
        keys[offset+0] = cents[0]
        keys[offset+2] = cents[1]
        keys[offset+4] = cents[2]
        keys[offset+5] = cents[3]
        keys[offset+7] = cents[4]
        keys[offset+9] = cents[5]
        keys[offset+11] = cents[6]

        keys[offset+1] = 0.5*(keys[offset+0]+keys[offset+2])
        keys[offset+3] = 0.5*(keys[offset+2]+keys[offset+4])
        keys[offset+6] = 0.5*(keys[offset+5]+keys[offset+7])
        keys[offset+8] = 0.5*(keys[offset+7]+keys[offset+9])
        keys[offset+10] = 0.5*(keys[offset+9]+keys[offset+11])

        for i in range(0,offset):
            keys[i] = keys[i+12]-1200
        for i in range(offset+12,128):
            keys[i] = 1200+keys[i-12]
        for i in range(0,128):
            keys[i]+=offset*100
        return keys


    def export_tun(self,fn,offset=0,add_scale_to_filename=True):
        keys = self.get_keys(offset)

        rep = self.print_report(disp=False)
        

        strr=""
        strr += "; Major {} Scale\n".format(self.offsets[offset])

        tmpstr = "[ "
        for i in range(0,len(self.x)-1):
            tmpstr += "{0:.02f}, ".format(self.x[i])
        tmpstr += "{0:.02f} ]".format(self.x[-1])
        strr+= "; Scale Cents = " + tmpstr+ "\n"

        tmpstr = "[ "
        for i in range(0,len(self.res["x"])-1):
            tmpstr += "{0:.02f}, ".format(self.res["x"][i])
        tmpstr += "{0:.02f} ]".format(self.res["x"][-1])
        strr+= "; Scale Steps = " + tmpstr+ "\n"
        strr+= ";\n;Tuning Report - Just Intonation errors (in Cents) :\n;"
        strr+= "\n;".join(rep)


        strr += '''
[Scale Begin]
Format= "AnaMark-TUN"
FormatVersion= 200
FormatSpecs= "http:\\\\www.mark-henning.de\\eternity\\tuningspecs.html"

[Info]
Name = "Diatonic Custom scale {0}"
Id = "ID_diatonic_custom_{0}"
'''.format(self.offsets[offset])
        strr+='''
[Exact Tuning]
basefreq=8.17579891564371
'''

        for i in range(0,128):
            strr+="note {0} = {1:.04f}\n".format(i,keys[i])
        strr+= '''
[Scale End]
'''
        if add_scale_to_filename:
            if fn[-4::] == ".tun":
                fn = fn[0:-4] + "_"+self.offsets[offset]+".tun"
            else:
                fn = fn + "_"+self.offsets[offset]+".tun"
        tun_strr_ascii = strr.encode('ascii')
        with open(fn, 'wb') as f:
            f.write(tun_strr_ascii)



if __name__=="__main__":
    config=[
        {
            "name":"Minor 2nd",
            "matrix":np.array(
            [[0,0,1,0,0,0,0],
            [0,0,0,0,0,0,1]]
            ),
            "weight":None,
            "target":ratio_to_cents(16/15),
            "semis":1
        },
        {
            "name":"Major 2nd",
            "matrix":np.array(
                [[1,0,0,0,0,0,0],
                [0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0],
                [0,0,0,0,0,1,0]
                ]
            ),
            "weight":None,
            "target":ratio_to_cents(9/8),
            "semis":2
        },
        {
            "name":"Minor 3rd",
            "matrix":np.array(
                [[0,1,1,0,0,0,0],
                [0,0,1,1,0,0,0],
                [0,0,0,0,0,1,1],
                [1,0,0,0,0,0,1]
                ]
            ),
            "weight":1.0*np.ones(4),
            "target":ratio_to_cents(6/5),
            "semis":3
        },
        {
            "name":"Major 3rd",
            "matrix":np.array(
                [[1,1,0,0,0,0,0],
                [0,0,0,1,1,0,0],
                [0,0,0,0,1,1,0]
                ]
            ),
            "weight":1.0*np.ones(3),
            "target":ratio_to_cents(5/4),
            "semis":4
        },
        
        {
        "name":"Perfect 4th",
        "matrix":np.array([
            [1,1,1,0,0,0,0],
            [0,1,1,1,0,0,0],
            [0,0,1,1,1,0,0],
            [0,0,0,0,1,1,1],
            [1,0,0,0,0,1,1],
        ]),
        "weight":None,
        "target":ratio_to_cents(4/3),
        "semis":5
        },
        {
            "name":"Perfect 5th",
            "matrix":np.array([
                [1,1,1,1,0,0,0],
                [0,1,1,1,1,0,0],
                [0,0,1,1,1,1,0],
                [0,0,0,1,1,1,1],
                [1,0,0,0,1,1,1],
                [1,1,0,0,0,1,1]
            ]),
            "weight":3.0*np.ones(6),
            "target":ratio_to_cents(3/2),
            "semis":7
        },
        {
            "name":"Octave",
            "matrix":np.ones((7,7)),
            "weight":np.ones(7)*10.0,
            "target":ratio_to_cents(2),
            "semis":12
        }
    ]

    
    sg = MajorScaleGenerator(config,octave_locked=True)
    sg.optimize()
    sg.print_report()
    for i in range(0,12):
        tun_strr = sg.export_tun('output_files/major_scale_optimized.tun',offset=i,add_scale_to_filename=True)
    