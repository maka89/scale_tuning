
import numpy as np
def ratio_to_cents(x):
    return 1200*np.log2(x)
def cents_to_ratio(x):
    return 2**(x/1200)

class JustGenerator:
    def __init__(self):
        self.ratios = [1.0,16/15,9/8,6/5,5/4,4/3,7/5,3/2,8/5,5/3,16/9,15/8,2]
        self.cents = [ ratio_to_cents(x) for x in self.ratios]
        self.offsets = {0:"C",1:"Db",2:"D",3:"Eb",4:"E",5:"F",6:"Gb",7:"G",8:"Ab",9:"A",10:"Bb",11:"B"}
    def get_keys_just(self,offset=0,in_tune_key=None):
        assert(offset<12)
        if in_tune_key:
            assert(in_tune_key < 12)
        keys = {}
        
        for i in range(0,12):
            keys[offset+i] = self.cents[i]

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
        keys = self.get_keys_just(offset,in_tune_key=in_tune_key)

        
        strr=""
        strr += ";Just Intonation {}\n".format(self.offsets[offset])
        if in_tune_key is not None:
            strr += ";{} In Perfect tune \n".format(self.offsets[in_tune_key])
        else:
            strr += ";{} In Perfect tune \n".format(self.offsets[offset])

        tmpstr = "[ "
        for i in range(0,len(self.cents)-1):
            tmpstr += "{0:.02f}, ".format(self.cents[i])
        tmpstr += "{0:.02f} ]".format(self.cents[-1])
        strr+= "; Scale Cents = " + tmpstr+ "\n"

        dc = np.array(self.cents)[1::]-np.array(self.cents)[0:-1]
        tmpstr = "[ "
        for i in range(0,len(dc)-1):
            tmpstr += "{0:.02f}, ".format(dc[i])
        tmpstr += "{0:.02f} ]".format(dc[-1])
        strr+= "; Scale Steps = " + tmpstr+ "\n"

        if in_tune_key is not None:
            tuned_key = self.offsets[in_tune_key]
        else:
            tuned_key = self.offsets[offset]
        strr += '''
[Scale Begin]
Format= "AnaMark-TUN"
FormatVersion= 200
FormatSpecs= "http:\\\\www.mark-henning.de\\eternity\\tuningspecs.html"

[Info]
Name = "Just Intonation {0} tuned for {1}"
Id = "just_intonation_{0}_tuned_{1}"
'''.format(self.offsets[offset],tuned_key)
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

if __name__ == "__main__":
    sg = JustGenerator()
    for i in range(0,12):
        for j in range(0,12):
            tun_strr = sg.export_tun('output_files/just_intonation/just.tun',offset=i,in_tune_key=j,add_scale_to_filename=True)

