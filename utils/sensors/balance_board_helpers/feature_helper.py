from scipy.spatial import ConvexHull
import numpy as np
from scipy.integrate import simpson
from scipy import signal
import antropy as ant
import scipy.stats
import nolds
from utils.sensors.balance_board_helpers import diffusion_stabilogram
from utils.sensors.balance_board_helpers import recurrence_quantification_analysis
from utils.sensors.balance_board_helpers import fractal_dimension

## NOTE: Recordings from Bertec Acquire have the following specifications:
#Row 1 = Time
#Row 2 = Fz
#Row 3 = Mx
#Row 4 = My
#Row 5 = CoPx = CoP_ML
#Row 6 = CoPy = CoP_AP
#Note. CoPx = -My/Fz
#Note. CoPy = Mx/Fz



def _recenter(data):
    """De-means the data"""
    data = np.array(data)
    return data - data.mean()

def _delta(data):
    """Gets the difference in data, i.e., delta[i] = x[i+1] - x[i]"""
    d1 = np.array(data[:-1])
    d2 = np.array(data[1:])
    return d2 - d1


def _eig(data):
    """Returns eigenvectors and eigenvalues from the x y"""

def _confidence_ellipse(x,y):
    N = len(x)
    corr = np.zeros([2,2])
    corr[0,0] = sum(x ** 2)
    corr[1,1] = sum(y ** 2)
    corr[0,1] = corr[1,0] = sum(x * y)
    w,v = np.linalg.eig(corr)
        
    major_idx = np.argmax(w)
    minor_idx = np.argmin(w)

    major_radius = np.sqrt(w[major_idx]/(N-1))
    minor_radius = np.sqrt(w[minor_idx]/(N-1))
    major_axis=v[:,major_idx]
    minor_axis=v[:,minor_idx]
    
    return major_radius,minor_radius,major_axis,minor_axis

def _get_psd(data,method=None):
    
    T = data[0][1] - data[0][0]
    fs = 1/T
    
    if method == 'multitaper':
        from mne.time_frequency import psd_array_multitaper
        psd_ML, f_ML = psd_array_multitaper(data[4], fs, adaptive=True, normalization='full', verbose=0)
        psd_AP, f_AP = psd_array_multitaper(data[5], fs, adaptive=True, normalization='full', verbose=0)
    elif method == None:
        f_ML, psd_ML = signal.periodogram(data[4], fs=fs)
        f_AP, psd_AP = signal.periodogram(data[5], fs=fs)    
    else:
        print("Please enter a valid method. Either 'multitaper' or None")
        return
    
    return psd_ML, psd_AP, f_ML, f_AP

####################################

def get_area95(data):
    """following https://www1.udel.edu/biology/rosewc/kaap686/reserve/cop/center%20of%20position%20conf95.pdf """
    
    x, y = _recenter(data[4]), _recenter(data[5])
    major_radius,minor_radius,_,_ = _confidence_ellipse(x, y)
    
    area95 = 5.991 * np.pi * major_radius * minor_radius
    return area95

def get_swayarea(data):
    """Returns sway area of the stabilogram. Defined by the convex hull of all points"""
    cop_x = data[4]
    cop_y = data[5]
    return ConvexHull(list(zip(cop_x,cop_y))).volume #Volume = area for a 2d shape 

def get_area95majoraxis(data):
    """Returns the angle of the major axis wrt the x axis, from the area95 ellipse"""

    x, y = _recenter(data[4]), _recenter(data[5])
    _,_,major_axis,minor_axis = _confidence_ellipse(x, y)
        
    vector_1 = [1,0] #x axis
    vector_2 = major_axis
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.degrees(np.arccos(dot_product))
    return angle
    

def get_area95_axis_length(data):
    """Returns the major and minor axis lengths from the area95 ellipse"""
    
    x, y = _recenter(data[4]), _recenter(data[5])
    major_radius,minor_radius,_,_ = _confidence_ellipse(x, y)
    
    major_axis_length = np.sqrt(5.991)*major_radius*2
    minor_axis_length = np.sqrt(5.991)*minor_radius*2
    return major_axis_length, minor_axis_length


def get_area95_minoraxis_tangent(data):
    """Is extremely poorly defined in doi: 10.1002/mds.25449. Is left blank here"""
    return None


def get_markedarea(data):
    """The calculation of the surface is carried out graphically with a res-
    olution of 0.0025 cm 2 . 
    Continuous triangles from the mean
    value of all measurement values to the last measurement point
    to the current measurement point are calculated. Points on the
    grid which overlap numerous times are not counted more than
    once (measured in square meters).
    
    POORLY DEFINED. Is this alpha shape or something?
    """
    return


def get_area90_length(data):
    """Is very poorly defined in the corresponding paper (doi: 10.1123/mcj.6.3.246). We are assuming that this is simply the 90% confidence interval in ML and AP directions"""
    
    x, y = _recenter(data[4]), _recenter(data[5])
    
    confidence = 0.9
    
    n = len(x)
    std_err_x = scipy.stats.sem(x)
    interval_x = std_err_x * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    CI_90_ML = interval_x * 2
    
    n = len(y)
    std_err_y = scipy.stats.sem(y)
    interval_y = std_err_y * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    CI_90_AP = interval_y * 2
    
    return CI_90_ML, CI_90_AP
    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
    
    
    major_radius,minor_radius,_,_ = _confidence_ellipse(x, y)
    
    major_axis_length = np.sqrt(4.605)*major_radius*2
    minor_axis_length = np.sqrt(4.605)*minor_radius*2
    
    return major_axis_length, minor_axis_length
    
    
def get_pathlength(data):
    """Returns pathlength, as well as pathlength in the ML and AP direction"""
    
    path_ML, path_AP = _delta(data[4]), _delta(data[5])
    pathlength_ML = sum(abs(path_ML))
    pathlength_AP = sum(abs(path_AP))
    pathlength = sum(np.sqrt(path_ML ** 2 + path_AP ** 2)) #Euclidean distances
    
    return pathlength, pathlength_ML, pathlength_AP    
    
    
def get_rms_displacement(data):
    """Returns the root mean square of radial displacement relative to center, also in the ML and AP direction"""
        
    displacements_ML, displacements_AP = _recenter(data[4]), _recenter(data[5])
    displacements = np.sqrt(displacements_ML**2 + displacements_AP**2)
    
    def _get_rms(data):
        return np.sqrt((data ** 2).mean())
    
    return _get_rms(displacements), _get_rms(displacements_ML), _get_rms(displacements_AP)    
    
    
def get_stdev_displacement(data):
    """Returns the standard deviation of radial displacement relative to center, also in the ML and AP direction"""
    
    displacements_ML, displacements_AP = _recenter(data[4]), _recenter(data[5])
    displacements = np.sqrt(displacements_ML**2 + displacements_AP**2)
    
    return displacements.std(), displacements_ML.std(), displacements_AP.std() 
    
    
    
    
def get_average_displacement(data):
    """Returns the average displacement, the mean of radial displacement from center"""
    
    displacements_ML, displacements_AP = _recenter(data[4]), _recenter(data[5])
    displacements = np.sqrt(displacements_ML ** 2 + displacements_AP ** 2) #Euclidean distances
    return displacements.mean()
    
    
    
def get_average_displacement_directional(data):
    """Returns the average displacement in AP/ML direction, different from get_average_displacement. This is the mean of raw displacement in the AP and ML direction, i.e., the center"""
    
    displacements_ML, displacements_AP = data[4], data[5]    
    return displacements_ML.mean(), displacements_AP.mean()
    
    
        
def get_displacement_range(data):
    """Returns the range of displacement in the ML and AP direction"""
        
    displacements_ML, displacements_AP = data[4], data[5]   
    
    displacement_range_ML = max(displacements_ML) - min(displacements_ML)
    displacement_range_AP = max(displacements_AP) - min(displacements_AP)
    
    return displacement_range_ML, displacement_range_AP
    
        

def get_peak_displacements(data):
    """Returns the peak displacement in ML, AP, forward, backward, left, and right, relative to center of sway"""
        
    displacements_ML, displacements_AP = _recenter(data[4]), _recenter(data[5])
    
    peak_forward = abs(displacements_AP.max()) #assumes forward is postive
    peak_backward = abs(displacements_AP.min())
    peak_left = abs(displacements_ML.min()) #assumes left is negative
    peak_right = abs(displacements_ML.max())
    peak_AP = max(peak_forward,peak_backward)
    peak_ML = max(peak_left,peak_right)
    
    return peak_ML, peak_AP, peak_forward, peak_backward, peak_left, peak_right
        
def get_direction_index(data):
    """Returns the direction index in the ML and AP direction"""
    pathlength,pathlength_ML,pathlength_AP = get_pathlength(data)
    DI_AP = pathlength_AP/pathlength
    DI_ML = pathlength_ML/pathlength
    
    return DI_ML, DI_AP
    
    
def get_swayratio(data):
    """Uses a Chebychev II 10th order low pass filter of 0.4hz to get CoM from CoP.
    Determine a good rs, that is, the minimum attenuation required in the stop band. 20 is standard"""
    
    _, pathlength_ML, pathlength_AP = get_pathlength(data)
    
    t = data[0]
    T = t[1] - t[0]
    fs = 1/T
    cutoff = 0.4
    
    sos = signal.cheby2(10, rs=20, Wn=cutoff, btype='lowpass',output='sos', fs=fs) #Chebyshev II 10th order low pass filter with cutoff at 0.4Hz and 20dB of stopband attenuation
    
    CoM_ML = signal.sosfilt(sos,data[4])
    CoM_AP = signal.sosfilt(sos,data[5])
    
    CoM_data = (t,data[1],data[2],data[3],CoM_ML,CoM_AP)
    _, CoM_pathlength_ML, CoM_pathlength_AP = get_pathlength(CoM_data)
    
    swayratio_ML = pathlength_ML/CoM_pathlength_ML
    swayratio_AP = pathlength_AP/CoM_pathlength_AP
    
    return swayratio_ML, swayratio_AP

def get_swaymovement(data):
    """Returns the sway movement (avg displacement from first half to last half), also in the ML and AP direction.
    Originally defined for a 60 second recording, so we will do the same (i.e., 30 seconds each), even though our recordings are actually 120 seconds."""
    
    t,x,y = np.array(data[0]),np.array(data[4]),np.array(data[5])
    
    mid_idx = int(len(t)/2)
    
    swaymovement_ML = x[:mid_idx].mean() - x[mid_idx:].mean()
    swaymovement_AP = y[:mid_idx].mean() - y[mid_idx:].mean()
    
    swaymovement = np.sqrt(swaymovement_ML ** 2 + swaymovement_AP ** 2)
    return swaymovement, swaymovement_ML, swaymovement_AP
    
    
def get_equilibriumscore(data):
    """Says it can be retrieved from only a forceplate, but I dont think so"""
    return
    
def get_surfacelengthratio(data):
    """The ratio between pathlength and area95"""
    
    area95 = get_area95(data)
    pathlength, _, _ = get_pathlength(data)
    
    return pathlength/area95
    
    
def get_planardeviation(data):
    """The sqrt of sum of varaince of displacements in the AP and ML direction"""
    
    _, stdev_displacement_ML, stdev_displacement_AP = get_stdev_displacement(data)
    planardeviation = np.sqrt(stdev_displacement_ML ** 2 + stdev_displacement_AP ** 2)
    
    return planardeviation
    
    
def get_phaseplaneparameter(data):
    """The sqrt of sum of variance of displacements (in x and y) and velocities (in x and y). Velocities are defined by instantaneous velocities."""
    
    _, stdev_displacement_ML, stdev_displacement_AP = get_stdev_displacement(data)
        
    path_ML, path_AP, t = _delta(data[4]), _delta(data[5]), _delta(data[0])
    instantaneous_velocities_ML = path_ML/t
    instantaneous_velocities_AP = path_AP/t
    
    phaseplaneparameter = np.sqrt(stdev_displacement_ML ** 2 + stdev_displacement_AP ** 2 + instantaneous_velocities_ML.std() ** 2 + instantaneous_velocities_AP.std() ** 2)
    
    return phaseplaneparameter

def get_average_velocity(data):
    """The average velocity defined as pathlength over time. Note that one way velocity can be defined is directional (i.e., positive and negative) and an average results in something close to 0.
    We do not use this definition as the other papers use the pathlength over time definition, which is really average speed, rather than average velocity (as velocity implies direction)"""
    
    pathlength, pathlength_ML, pathlength_AP = get_pathlength(data)
    t = data[0]
    
    average_velocity = pathlength/max(t)
    average_velocity_ML = pathlength_ML/max(t)
    average_velocity_AP = pathlength_AP/max(t)
      
    return average_velocity, average_velocity_ML, average_velocity_AP
    
def get_peak_velocities(data):
    """Returns peak instantaneous velocity in a given direction"""
    
    path_ML, path_AP, t = _delta(data[4]), _delta(data[5]), _delta(data[0])
    path = np.sqrt(path_ML ** 2 + path_AP ** 2) #Euclidean distances
    
    instantaneous_velocities = path/t
    instantaneous_velocities_ML = path_ML/t
    instantaneous_velocities_AP = path_AP/t
    
    peak_velocity_left = abs(min(instantaneous_velocities_ML)) #assumes left is negative
    peak_velocity_right = abs(max(instantaneous_velocities_ML))
    peak_velocity_forward = abs(max(instantaneous_velocities_AP))
    peak_velocity_backward = abs(min(instantaneous_velocities_AP)) #assumes backward is negative
    
    return peak_velocity_forward, peak_velocity_backward, peak_velocity_left, peak_velocity_right 
    
        
def get_bandpowers(data,lower_band,upper_band, method=None, relative=False):
    """Returns the bandpower for given frequencies in the ML and AP direction"""
    
    psd_ML, psd_AP, f_ML, f_AP = _get_psd(data,method)
    
    freq_res_ML = f_ML[1] - f_ML[0]
    freq_res_AP = f_AP[1] - f_AP[0] #Should be the same as above, but just in case
    
    idx_band_ML = np.logical_and(f_ML >= lower_band, f_ML <= upper_band)    # Find index of band in frequency vector
    idx_band_AP = np.logical_and(f_AP >= lower_band, f_AP <= upper_band)    # Find index of band in frequency vector
    bp_ML = simpson(psd_ML[idx_band_ML], dx=freq_res_ML)    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp_AP = simpson(psd_AP[idx_band_AP], dx=freq_res_AP)

    if relative:
        bp_ML /= simpson(psd_ML, dx=freq_res_ML)
        bp_AP /= simpson(psd_AP, dx=freq_res_AP)
    
    return bp_ML, bp_AP
    
def get_edgefrequency(data, power_edge, method=None):
    """Returns the spectral edge frequency for a given power in the ML and AP direction. Power_edge should be a fraction"""
    

    psd_ML, psd_AP, f_ML, f_AP = _get_psd(data,method)
    
    freq_res_ML = f_ML[1] - f_ML[0]
    freq_res_AP = f_AP[1] - f_AP[0] #Should be the same as above, but just in case
        
    tot_power_ML = simpson(psd_ML, dx=freq_res_ML)
    tot_power_AP = simpson(psd_AP, dx=freq_res_AP)

    for i in range(len(psd_ML)):
        band_power =  simpson(psd_ML[:i+1], dx=freq_res_ML)
        power = band_power / tot_power_ML
        if power >= power_edge:
            edge_frequency_ML = f_ML[i]
            break
            
    for i in range(len(psd_ML)):
        band_power =  simpson(psd_AP[:i+1], dx=freq_res_AP)
        power = band_power / tot_power_AP
        if power >= power_edge:
            edge_frequency_AP = f_AP[i]
            break

    return edge_frequency_ML, edge_frequency_AP
    
    
    
def get_frequency95(data):
    """This is different from above as it is for the bidirectional case. Although the methods section implies that they are defined separately, the results imply that are somehow combined. 
    
    POORLY DEFINED"""
    return
    
    
def get_totalenergy(data, demean=True):
    """The integral of the energy spectral density of the sway in the AP and ML directions. For a discrete signal, this is simply the sum of squares of each point"""
    
    if demean: x, y = _recenter(data[4]), _recenter(data[5])
    else: x, y = np.array(data[4]), np.array(data[5])
    totalenergy_ML = sum(x ** 2)
    totalenergy_AP = sum(y ** 2)
    
    return totalenergy_ML, totalenergy_AP
    
    
    
    
    
##################### DIFFUSION PLOT ANALYSIS #####################

def get_diffusion_plot_analysis_features(data):
    """Returns all the features associated with the diffusion plot analysis, that is: DTXC, DTYC, DTRC, X2, Y2, R2, DXS, DYS, DRS, HXS, HYS, HRS, DXL, DYL, DRL, HXL, HYL, HRL"""
    
    DTXC, DTYC, DTRC, X2, Y2, R2, DXS, DYS, DRS, HXS, HYS, HRS, DXL, DYL, DRL, HXL, HYL, HRL = diffusion_stabilogram.get_diffusion_stabilogram_features(data)
    
    return DTXC, DTYC, DTRC, X2, Y2, R2, DXS, DYS, DRS, HXS, HYS, HRS, DXL, DYL, DRL, HXL, HYL, HRL
###################################################################


def get_fractaldimension(data):
    """This returns the fractal dimension. The original article uses the modified pixel dilation method, but instead, we opt for Minkowski-Bouligand dimension (or box counting dimension).
    
    There is a problem here that size of the sway affects the fractal dimension. Thus all the sways must be on the same scale to be comparable. However, some sways are very large relative to other, and thus by fitting the entire trajectory of all sways means that some sways will be reduced to filling a small percentage of the space provided. To remedy this to some degree, the most extreme 5% of points in both the AP and ML directions will be cut off. This is not to mean that everyones trajectories will be affected, but rather that several sways will be affected, whilst some not at all.
    """
    
    #These have been defined in the fractal dimension dev notebook, with demeaned data, and in the mm scale (i.e., scaled by 1000)
    lim = 33 #This retains at least 99% of all points. i.e., it removes at most 1% of the extreme points
    x_limits = (-lim, lim) 
    y_limits = (-lim, lim) #ensure the image created is a square
    
    f = fractal_dimension.get_fractal_dimension(data,x_limits,y_limits)
    
    return f


def get_swayvectorlength(data):
    """Return the vector length, equal to the mean CoP velocity"""
    
    average_velocity, _, _ = get_average_velocity(data)
    
    return average_velocity
    
    
def get_swayvectorangle(data):
    """Return the sway vector angle, equal to arctangent (pathlength AP /pathlength ML )"""
    
    _, pathlength_ML, pathlength_AP = get_pathlength(data)
    
    return np.degrees(np.arctan(pathlength_AP/pathlength_ML))
    
    
    
def get_averageradius(data):
    """The average radius of CoP points"""
    x, y = _recenter(data[4]), _recenter(data[5])
    radius = np.sqrt(x ** 2 + y ** 2)

    return radius.mean()
    
    
def get_covariance(data):
    """The covariance between the ML and AP directional aspects of sway. Poorly defined in text so assumed to be this"""
    x, y = _recenter(data[4]), _recenter(data[5])
    return np.cov(x,y)[0,1]
    
    
def get_sampleentropy(data):
    """Sample entropy, using antropy.sample_entropy, in the ML and AP directions. """
    
    x, y = np.array(data[4]), np.array(data[5])
    
    sample_entropy_ML = ant.sample_entropy(x)
    sample_entropy_AP = ant.sample_entropy(y)
    
    return sample_entropy_ML, sample_entropy_AP
    
def get_DLE(data):
    """This returns the dominant lyapunov exponent. This again is poorly defined in the corresponding paper, as it is usually for a single time series (i.e., AP or ML of CoP). We will split this."""

    t = data[0]
    T = t[1]-t[0]
    fs = int(np.round(1/T))    
    
    if fs == 10: tau=3
    elif fs == 20: tau=6
    elif fs == 40: tau=12
    elif fs == 100: tau=30
    else: 
        print('Dont recognise the fs')
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    
        DLE_ML = nolds.lyap_r(data[4], emb_dim=5, lag=tau)
        DLE_AP = nolds.lyap_r(data[5], emb_dim=5, lag=tau)
    
    return DLE_ML, DLE_AP
   
    
def get_RQA_features(data):
    """This returns the RQA features for the AP and ML direction. Specifically, these features are: % recurrance, % determinism, RQA entropy, maxline, and trend"""
    
    t = data[0]
    T = t[1]-t[0]
    fs = int(np.round(1/T))    
    
    if fs == 10: tau=3
    elif fs == 20: tau=6
    elif fs == 40: tau=12
    elif fs == 100: tau=30
    else: 
        print('Dont recognise the fs')
        return
    
    recurrence_ML, determinism_ML, entropy_ML, maxline_ML, trend_ML, recurrence_AP, determinism_AP, entropy_AP, maxline_AP, trend_AP = recurrence_quantification_analysis.get_RQA_features(data, m=5, tau=tau)
    
    #NOTE: The order here is different from above
    return recurrence_ML, recurrence_AP, determinism_ML, determinism_AP, entropy_ML, entropy_AP, maxline_ML, maxline_AP, trend_ML, trend_AP

    
####################### EO/EO ####################
    
    
def get_swayarea_romberg(eo_data,ec_data):
    """The romberg ratio (eyes closed divide by eyes open) of swayarea"""
    eo_swayarea = get_swayarea(eo_data)
    ec_swayarea = get_swayarea(ec_data)
    
    return ec_swayarea/eo_swayarea


def get_swayarea_vri(eo_data,ec_data):
    """VRI is the inverse of romberg ratio, i.e., is eyes open divide by eyes closed"""
    swayarea_romberg = get_swayarea_romberg(eo_data, ec_data)
    
    return 1/swayarea_romberg
    

def get_pathlength_romberg(eo_data,ec_data):
    """The romberg ratio (eyes closed divide by eyes open) of pathlength"""
    eo_pathlength, eo_pathlength_ML, eo_pathlength_AP = get_pathlength(eo_data)
    ec_pathlength, ec_pathlength_ML, ec_pathlength_AP = get_pathlength(ec_data)
    
    return ec_pathlength/eo_pathlength, ec_pathlength_ML/eo_pathlength_ML, ec_pathlength_AP/eo_pathlength_AP
    
    
def get_rms_displacement_vri(eo_data,ec_data):
    """VRI is the inverse of romberg ratio, i.e., is eyes open divide by eyes closed"""
    eo_rms_displacement, eo_rms_displacement_ML, eo_rms_displacement_AP = get_rms_displacement(eo_data)
    ec_rms_displacement, ec_rms_displacement_ML, ec_rms_displacement_AP = get_rms_displacement(ec_data)

    return eo_rms_displacement/ec_rms_displacement, eo_rms_displacement_ML/ec_rms_displacement_ML, eo_rms_displacement_AP/ec_rms_displacement_AP 

    
def get_pathlength_vri(eo_data,ec_data):
    """VRI is the inverse of romberg ratio, i.e., is eyes open divide by eyes closed"""
    pathlength_romberg, pathlength_ML_romberg, pathlength_AP_romberg = get_pathlength_romberg(eo_data,ec_data)
    return 1/pathlength_romberg, 1/pathlength_ML_romberg, 1/pathlength_AP_romberg
    
    
def get_average_displacment_vri(eo_data,ec_data):
    """VRI is the inverse of romberg ratio, i.e., is eyes open divide by eyes closed"""
    eo_average_displacement_ML, eo_average_displacement_AP = get_average_displacement_directional(eo_data)
    ec_average_displacement_ML, ec_average_displacement_AP = get_average_displacement_directional(ec_data)
    
    return eo_average_displacement_ML/ec_average_displacement_ML, eo_average_displacement_AP/ec_average_displacement_AP
    
     
     
