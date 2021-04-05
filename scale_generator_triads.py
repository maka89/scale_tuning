#import numpy as np
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
def ratio_to_cents(x):
    return 1200*np.log2(x)
def cents_to_ratio(x):
    return 2**(x/1200)

class ScaleGeneratorTriads:
    def __init__(self,config,octave_locked=True,order=2,n_iter=1):
        self.config=config
        self.out = None
        self.octave_locked = octave_locked
        self.order=order
        self.n_iter=n_iter
        self.offsets = {0:"C",1:"Db",2:"D",3:"Eb",4:"E",5:"F",6:"Gb",7:"G",8:"Ab",9:"A",10:"Bb",11:"B"}
    def optimize(self):

        def fn(x,args):
            retval = 0.0
            if self.octave_locked:
                #retval += 0.01*x[-1]**2
                x=np.concatenate((x[0:-1], np.array([1200-np.sum(x[0:-1])])))
                 
            conf = args[0]
            
            for c in conf:
                if c["weight"] is not None:
                    retval += np.mean( c["weight"]*np.abs((np.dot(c["matrix"],x)-c["target"])/5.0)**self.order )
                if c["weight_et"] is not None:
                    retval += np.mean( c["weight_et"]*np.abs((np.dot(c["matrix"],x)-c["semis"]*100)/5.0)**self.order )

            
            f0 = 1
            f1 = cents_to_ratio(np.sum(x[0:1]))
            f2 = cents_to_ratio(np.sum(x[0:2]))
            f4 = cents_to_ratio(np.sum(x[0:4]))

            f3 = cents_to_ratio(np.sum(x[0:3]))
            f5 = cents_to_ratio(np.sum(x[0:5]))
            f6 = cents_to_ratio(np.sum(x[0:6]))

            retval += 1000*(((f4-f2) - (f2-f0))/f0 )**2 #C
            retval += 1000*(((2*f0-f5) - (f5-f3))/f3)**2 #F
            retval += 1000*(((2*f1-f6) - (f6-f4))/f4)**2 #G

            return retval
        grad_fn = grad(fn,0)
        x0= np.array([200,200,100,200,200,200,100])

        best_fun = 1e60
        self.res = None
        for i in range(0,self.n_iter):
            xi = x0 + np.random.randn(7)*5.0
            if self.octave_locked:
                xi[-1] = 1200-np.sum(xi[0:-1])
            res = minimize(fn,x0=xi,jac=grad_fn,args=[self.config],method="l-bfgs-b",options={"ftol":1e-12,"gtol":1e-12})
            if res["fun"] < best_fun:
                self.res = res
                best_fun = res["fun"]


        if self.octave_locked:
            self.res["x"][-1] = 1200-np.sum(self.res["x"][0:-1])
        tmp = np.cumsum(self.res["x"])
        self.x = np.zeros(7)
        for i in range(1,7):
            self.x[i] = tmp[i-1]
        print(self.res)
        x=self.res["x"]
        f0 = 1
        f1 = cents_to_ratio(np.sum(x[0:1]))
        f2 = cents_to_ratio(np.sum(x[0:2]))
        f4 = cents_to_ratio(np.sum(x[0:4]))

        f3 = cents_to_ratio(np.sum(x[0:3]))
        f5 = cents_to_ratio(np.sum(x[0:5]))
        f6 = cents_to_ratio(np.sum(x[0:6]))
        
        print("C: ", ((f4-f2) - (f2-f0))/f0) #C
        print("F: ",((2*f0-f5) - (f5-f3) )/f3) #F
        print("G: ",((2*f1-f6) - (f6-f4))/f4) #G

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


    def get_keys(self,offset=0,in_tune_key=None):
        assert(offset<12)
        if in_tune_key:
            assert(in_tune_key < 12)
        cents = self.x
        keys = {}

        #Keys in scale
        keys[offset+0] = cents[0]
        keys[offset+2] = cents[1]
        keys[offset+4] = cents[2]
        keys[offset+5] = cents[3]
        keys[offset+7] = cents[4]
        keys[offset+9] = cents[5]
        keys[offset+11] = cents[6]

        #Accidentals
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

        in_tune_cents = 1.0*keys[in_tune_key]
        if in_tune_key is not None:
            for i in range(0,128):
                keys[i] = keys[i]-in_tune_cents+100*in_tune_key

        return keys


    def export_tun(self,fn,offset=0,in_tune_key=None,add_scale_to_filename=True):
        keys = self.get_keys(offset,in_tune_key=in_tune_key)

        rep = self.print_report(disp=False)
        
        strr=""
        strr += ";{} Scale\n".format(self.offsets[offset])
        if in_tune_key is not None:
            strr += ";{} In Perfect tune \n".format(self.offsets[in_tune_key])
        else:
            strr += ";{} In Perfect tune \n".format(self.offsets[offset])


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
            add_strr = "_"+self.offsets[offset]
            if in_tune_key is not None:
                add_strr += "_{}_tuned".format(self.offsets[in_tune_key])

            if fn[-4::] == ".tun":
                fn = fn[0:-4] +add_strr+".tun"
            else:
                fn = fn + add_strr+".tun"
        tun_strr_ascii = strr.encode('ascii')
        with open(fn, 'wb') as f:
            f.write(tun_strr_ascii)



