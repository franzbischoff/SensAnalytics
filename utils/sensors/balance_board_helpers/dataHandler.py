import scipy.io as io
from os.path import join
from scipy import signal
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_sway_file(path, verbose=False):
    """Loads a postural sway matlab data file"""
    data = io.loadmat(path)['data']
    
    if verbose:
        #check sampling frequency
        t = data[0]
        T = t[1] - t[0]
        if 1/T != 1000:
            print("The sampling rate for %s does not seem to be 1000 Hz. It is actually %i" % (path, 1/T))
            
        #check length
        if len(t) != 120000:
            print("The sample (%s) does not seem to be 120 seconds. It is actually %i" % (path, len(t)/1000))

    return data
    
    
def data_summary(participants):
    
    c = df_retrieve(participants,{'is PD':False})
    pd = df_retrieve(participants,{'is PD':True})
    
    
    
    cf = df_retrieve(participants,{'is PD':False,'Sex':'Female'})
    cm = df_retrieve(participants,{'is PD':False,'Sex':'Male'})
    pf = df_retrieve(participants,{'is PD':True,'Sex':'Female'})
    pm = df_retrieve(participants,{'is PD':True,'Sex':'Male'})
    
    
    print("There are %i PD patients, %i males and %i females, with age %.2f (+- %.2f), UPDRS %.2f (+- %.2f), and yrs since diagnosis %.2f (+- %.2f)" % (len(pd), len(pm), len(pf), pd['Age'].mean(), pd['Age'].std(), pd['UPDRS'].mean(), pd['UPDRS'].std(), pd['Yrs since diagnosis'].mean(), pd['Yrs since diagnosis'].std()))
    print("\tThe %i males have age %.2f (+- %.2f), UPDRS %.2f (+- %.2f), and yrs since diagnosis %.2f (+- %.2f)" % (len(pm), pm['Age'].mean(), pm['Age'].std(), pm['UPDRS'].mean(), pm['UPDRS'].std(), pm['Yrs since diagnosis'].mean(), pm['Yrs since diagnosis'].std()))
    print("\tThe %i females have age %.2f (+- %.2f), UPDRS %.2f (+- %.2f), and yrs since diagnosis %.2f (+- %.2f)" % (len(pf), pf['Age'].mean(), pf['Age'].std(), pf['UPDRS'].mean(), pf['UPDRS'].std(), pf['Yrs since diagnosis'].mean(), pf['Yrs since diagnosis'].std()))
    
    print("There are %i control participants, %i males and %i females, with age %.2f (+- %.2f)" % (len(c), len(cm), len(cf), cf['Age'].mean(), cf['Age'].std()))
    print("\tThe %i males have age %.2f (+- %.2f)" % (len(cm), cm['Age'].mean(), cm['Age'].std()))
    print("\tThe %i females have age %.2f (+- %.2f)" % (len(cf), cf['Age'].mean(), cf['Age'].std()))



def _decimate(data,downsampling_factor):
    """The working portion of the decimate function. The downsampling_factor should be either an int of <= 10, or a list of integers <= 10, representing a cascading decimation"""
    
    if isinstance(downsampling_factor,int):
        if downsampling_factor > 10:
            print('Make sure the downsampling factor is less than 10. If you want it to be more than 10, cascade it and present it as a list. E.g., [10,10] gives a factor of 100')
            return
        else:
            data = signal.decimate(data,downsampling_factor)
    
    if isinstance(downsampling_factor,list):
        for factor in downsampling_factor:
            if factor > 10:
                print('Make sure the downsampling factor is less than 10. If you want it to be more than 10, cascade it and present it as a list. E.g., [10,10] gives a factor of 100')
                return
            else:    
                data = signal.decimate(data,factor)
                
    return data
 
 
 
def df_retrieve(data,cols_vals):
    """
    Retrieves a subset of a dataframe based on equivalence matches
    
    Parameters:
        data (pandas.DataFrame): the dataframe from which a subset is to be created from
        cols_vals (dict): a dict with the key as the column name and the val as the equivalence val
    """
    for col,val in cols_vals.items():
        data = data[data[col] == val]
    
    return data
 
 
    
def decimate(data,downsampling_factor):
    """Decimates (low pass filter and then downsamples) the signal. This decimates only the Fz, Mx, My, CoPx, and CoPy. That is, it does not decimate t, but rather simply downsamples it"""
    
    CoPx = _decimate(data[4],downsampling_factor)
    CoPy = _decimate(data[5],downsampling_factor)
    Fz = _decimate(data[1],downsampling_factor)
    Mx = _decimate(data[2],downsampling_factor)
    My = _decimate(data[3],downsampling_factor)
    
    t = []
    for i,ts in enumerate(data[0]): 
        if i % np.prod(downsampling_factor) == 0: t.append(ts)
    
    return np.array(t),Fz,Mx,My,CoPx,CoPy
    

def get_data_info(path):
    
    data = load_sway_file(path)
    if len(data) == 6:
        t, fz, mx, my, copx, copy = data
    else:
        fz, mx, my, copx, copy = [data[:,i] for i in range(data.shape[1])]
        t = [np.nan,np.nan]
    T = t[1] - t[0]
    fs = 1/T
    time = len(t)/fs
    
    samples=len(fz)
    copx_equation = all(copx == -my/fz)
    copy_equation = all(copy == mx/fz)
    weight = (fz/9.81).mean()
    
    return samples, fs, time, copx_equation, copy_equation, weight
    
def view_data(Fz, My, Mx, CoPx=None, CoPy=None,t=None, stabilogram_xlims=None, stabilogram_ylims=None):
    
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=[20,5])
    ax1.plot(Fz)
    ax1.set_title('Fz')
    ax2.plot(My, label='My')
    ax2.plot(Mx, label='Mx')
    ax2.legend(loc='upper right')
    ax2.set_title('My and Mx')
    
    
    ax3.set_title('CoPx, CoPy, -My/Fz, and Mx/Fz')
    if (CoPx is not None) and (CoPy is not None):
        ax3.plot(CoPx, label='CoPx',alpha=0.75, linewidth=6)
        ax3.plot(CoPy, label='CoPy',alpha=0.75, linewidth=6)
        
        ax4.set_title('Stabilogram of CoP')
        ax4.plot(CoPx, CoPy)
    else:
        ax4.set_title('Stabilogram of -My/Fz and Mx/Fz')
        ax4.plot(-My/Fz, Mx/Fz)
    
    ax3.plot(-My/Fz, label='-My/Fz',alpha=0.75)
    ax3.plot(Mx/Fz, label='Mx/Fz',alpha=0.75)
    ax3.legend(loc='upper right')
    if stabilogram_xlims is not None: ax4.set_xlim(stabilogram_xlims)
    if stabilogram_ylims is not None: ax4.set_ylim(stabilogram_ylims)
    
    ax4.set_xlabel('ML')
    ax4.set_ylabel('AP')
    
    plt.show()
    

def pickleSave(path,obj):
    with open(path,'wb') as f:
        pickle.dump(obj,f)
    return

def pickleLoad(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
    return obj


