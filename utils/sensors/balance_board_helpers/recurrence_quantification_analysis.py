from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
import numpy as np
from scipy.optimize import minimize_scalar


def get_RQA_features(data, m, tau):
    
    t, _, _, _, x, y = data
    T = t[1]-t[0]
    fs = int(np.round(1/T))    
    theiler_window = 1*fs #1 second
    
    recurrence_ML, determinism_ML, entropy_ML, maxline_ML, trend_ML = _get_RQA_features(x,m,tau,theiler_window)
    recurrence_AP, determinism_AP, entropy_AP, maxline_AP, trend_AP = _get_RQA_features(y,m,tau,theiler_window)
    
    return recurrence_ML, determinism_ML, entropy_ML, maxline_ML, trend_ML, recurrence_AP, determinism_AP, entropy_AP, maxline_AP, trend_AP
    
def _get_RQA_features(data, m, tau, theiler_window):
    
    time_series = TimeSeries(data, embedding_dimension=m, time_delay=tau)
    r = solve_for_radius(time_series,0.05,theiler_window)
    settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(r), similarity_measure=EuclideanMetric, theiler_corrector=theiler_window) #theiler_corrector = 1second
    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2
    
    recurrence = result.recurrence_rate
    determinism = result.determinism
    entropy = result.entropy_diagonal_lines
    maxline = result.longest_diagonal_line
    
    computation = RPComputation.create(settings)
    recurrence_plot = computation.run()
    trend, _, _, _ = get_trend(recurrence_plot.recurrence_matrix,theiler_window=theiler_window)
    
    return recurrence, determinism, entropy, maxline, trend
    
#In a recurrence plot, 1 represents a recurrence. In matrix form, top left is 0,0. In image form, bottom left is 0,0

def solve_for_radius(time_series,target_recurrence = 0.05, theiler_corrector=1):

    def to_minimise(r, time_series, target_recurrence, theiler_corrector=1):
        settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(r), similarity_measure=EuclideanMetric, theiler_corrector=theiler_corrector)
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
        result.min_diagonal_line_length = 2
        result.min_vertical_line_length = 2
        result.min_white_vertical_line_length = 2           
         
        return abs(target_recurrence-result.recurrence_rate)
        
    additional_args = (time_series,target_recurrence,theiler_corrector)
    res = minimize_scalar(to_minimise,args=additional_args)
    return res.x

    
    
    
    
def get_trend(matrix, theiler_window = 0):
    """ 
    Drift in a dynamical system is characterized by paling of the recurrence plot away from the central diagonal. 
    To quantify drift we compute the percentage of recurrent points in long diagonals parallel to the central line. 
    Percentage values are plotted as a function of distance away from the central diagonal, and a line of best fit is computed.
    Our trend value is defined as the slope of this line.
    
    Parameters:
        matrix: A recurrence matrix, 1 represents a recurrent point. In matrix form, top left is 0,0. In image form, bottom left is 0,0.
    """

    x = [] #distance away from centre
    y = [] #percentage values

    #We will only look at the upper triangular, as the lower triangular is mirrored
    for i in range(theiler_window,len(matrix)):
        recurrences = 0
        length = 0
        for j in range(0,len(matrix)-i):
            recurrences += matrix[j,j+i]
            length += 1
            
        percent_recurrence = recurrences/length
        
        x.append(i)
        y.append(percent_recurrence)    
    
    trend, intercept = np.polyfit(x,y,1)
    
    return trend, intercept, x, y
