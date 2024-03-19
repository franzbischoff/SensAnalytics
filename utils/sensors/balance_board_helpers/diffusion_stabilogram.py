import numpy as np
import plotly.graph_objects as go

## NOTE: Recordings from Bertec Acquire have the following specifications:
#Row 1 = Time
#Row 2 = Fz
#Row 3 = Mx
#Row 4 = My
#Row 5 = CoPx = CoP_ML
#Row 6 = CoPy = CoP_AP
#Note. CoPx = -My/Fz
#Note. CoPy = Mx/Fz


def _delta(data, degree=1):
    """Gets the difference in data, i.e., delta[i] = x[i+degree] - x[i]"""
    d2 = np.array(data[degree:])
    degree = len(data) - degree
    d1 = np.array(data[:degree])
    return d2 - d1
    
    
def get_diffusion_stabilogram_points(data,scaler=1):
    """Returns the diffusion stabilogram points for a given delta t, also in the ML and AP directions"""
    deltas, deltas_ML, deltas_AP = [], [], []
    times = []
    t,_,_,_,CoPx,CoPy = data
    CoPx *= scaler
    CoPy *= scaler
    max_d = np.where(t<11)[0][-1] #gets the index of the last time step where it is less than 11 seconds
    for d in range(max_d): #get the points for delta_t of up to 11 seconds. I know the line fitting occurs at 10 seconds, so we calculate a bit more just in case
        delta_CoPx = _delta(CoPx,d)
        delta_CoPy = _delta(CoPy,d)
        delta = np.sqrt(delta_CoPx**2 + delta_CoPy**2)
        
        deltas_ML.append((delta_CoPx**2).mean())
        deltas_AP.append((delta_CoPy**2).mean())
        deltas.append((delta**2).mean())
        times.append(t[d])
        
    return np.array(deltas), np.array(deltas_ML), np.array(deltas_AP), np.array(times)


def get_scaling_exponents(diffusion_stabilogram_points, time, short_term_window, long_term_window):
    """Get H, the scaling exponent of fractal Brownian motion, by finding the gradient in the short and long term windows on a log-log plot of the diffusion stabilogram.
    
    Parameters:
        diffusion_stabilogram_points (array-like): 
        time (array-like)
        short_term_window (tuple): of [lower_limit, upper_limit]
        long_term_window (tuple): of [lower_limit, upper_limit]
    """
    condition = (time > short_term_window[0]) & (time <= short_term_window[1]) #to exclude 0 if given as lower limit
    where = np.where(condition)
    short_term_points = diffusion_stabilogram_points[where]
    short_term_t = time[where]
    H_s, c_s = np.polyfit(np.log10(short_term_t),np.log10(short_term_points),1)
    
    condition = (time >= long_term_window[0]) & (time <= long_term_window[1])
    where = np.where(condition)
    long_term_points = diffusion_stabilogram_points[where]
    long_term_t = time[where]
    H_l, c_l = np.polyfit(np.log10(long_term_t),np.log10(long_term_points),1)
    
    
    return H_s, c_s, H_l, c_l
    
def get_diffusion_coefficients(diffusion_stabilogram_points, time, short_term_window, long_term_window):
    """Get D, the diffusion coefficient of classical random walks, by finding the gradient in the short and long term windows on a plot of the diffusion stabilogram.
    
    Parameters:
        diffusion_stabilogram_points (array-like): 
        time (array-like)
        short_term_window (tuple): of [lower_limit, upper_limit]
        long_term_window (tuple): of [lower_limit, upper_limit]
    """
    
    condition = (time >= short_term_window[0]) & (time <= short_term_window[1]) #to include 0 if given as lower limit
    where = np.where(condition)
    short_term_points = diffusion_stabilogram_points[where]
    short_term_t = time[where]
    D_s, c_s = np.polyfit(short_term_t,short_term_points,1)
    
    condition = (time >= long_term_window[0]) & (time <= long_term_window[1])
    where = np.where(condition)
    long_term_points = diffusion_stabilogram_points[where]
    long_term_t = time[where]
    D_l, c_l = np.polyfit(long_term_t,long_term_points,1)
    
    return D_s, c_s, D_l, c_l
    
    
def get_linear_intersection(m1,c1,m2,c2):
    """gets the intersection of 2 straight lines described by y=mx+c"""    
    
    x_intersect = (c1-c2)/(m2-m1)
    y_intersect = x_intersect*m1+c1
    
    return x_intersect, y_intersect
        
    
def get_diffusion_stabilogram_features(data):
    """Returns all diffusion stabilogram features: critical time, diffusion stabilogram value at critical time, short term diffusion coefficient, long term diffusion coefficient, short term scaling exponent, and long term scaling exponent, also for the ML and AP directions. Specifically, that is: DTXC, DTYC, DTRC, X2, Y2, R2, DXS, DYS, DRS, HXS, HYS, HRS, DXL, DYL, DRL, HXL, HYL, HRL
    
     The diffusion stabilogram dev notebook concludes that for the linear plot, the short term window should be 0 <= t <= 0.4, and the long term window should be  2 <= t <= 10. For the log-log plot, the short term window should be 0 < t <= 0.4, and the long term window should be 1 <= t <= 10
     """
    t = data[0]
    T = t[1] - t[0]
    stabilogram_points, stabilogram_points_ML, stabilogram_points_AP, times = get_diffusion_stabilogram_points(data, scaler=1)
    
    DRS, DRS_intersect, DRL, DRL_intersect = get_diffusion_coefficients(stabilogram_points,times,short_term_window=[0,0.5],long_term_window=[2,10])
    DTRC, _ = get_linear_intersection(DRS, DRS_intersect, DRL, DRL_intersect)
    if DTRC < 0: DTRC, R2 = np.nan, np.nan
    else: R2 = stabilogram_points[np.round(DTRC/T).astype(int)]
    HRS, HRS_intersect, HRL, HRL_intersect = get_scaling_exponents(stabilogram_points,times,short_term_window=[0,0.4],long_term_window=[1,10])
    
    DXS, DXS_intersect, DXL, DXL_intersect = get_diffusion_coefficients(stabilogram_points_ML,times,short_term_window=[0,0.5],long_term_window=[2,10])
    DTXC, _ = get_linear_intersection(DXS, DXS_intersect, DXL, DXL_intersect)
    if DTXC < 0: DTXC, X2 = np.nan, np.nan
    else: X2 = stabilogram_points[np.round(DTXC/T).astype(int)]
#    X2 = stabilogram_points[np.round(DTXC/T).astype(int)]
    HXS, HXS_intersect, HXL, HXL_intersect = get_scaling_exponents(stabilogram_points_ML,times,short_term_window=[0,0.4],long_term_window=[1,10])
    
    DYS, DYS_intersect, DYL, DYL_intersect = get_diffusion_coefficients(stabilogram_points_AP,times,short_term_window=[0,0.5],long_term_window=[2,10])
    DTYC, _ = get_linear_intersection(DYS, DYS_intersect, DYL, DYL_intersect)
    if DTYC < 0: DTYC, R2 = np.nan, np.nan
    else: Y2 = stabilogram_points[np.round(DTYC/T).astype(int)]
#    Y2 = stabilogram_points[np.round(DTYC/T).astype(int)]
    HYS, HYS_intersect, HYL, HYL_intersect = get_scaling_exponents(stabilogram_points_AP,times,short_term_window=[0,0.4],long_term_window=[1,10])
    
    return DTXC, DTYC, DTRC, X2, Y2, R2, DXS, DYS, DRS, HXS, HYS, HRS, DXL, DYL, DRL, HXL, HYL, HRL
    
def plot_diffusion_stabilogram(data):

    
    stabilogram_points, stabilogram_points_ML, stabilogram_points_AP, times = get_diffusion_stabilogram_points(data, scaler=1)

    DRS, DRS_intersect, DRL, DRL_intersect = get_diffusion_coefficients(stabilogram_points,times,short_term_window=[0,0.5],long_term_window=[2,10])
    DTRC, R2 = get_linear_intersection(DRS, DRS_intersect, DRL, DRL_intersect)
    HRS, HRS_intersect, HRL, HRL_intersect = get_scaling_exponents(stabilogram_points,times,short_term_window=[0,0.4],long_term_window=[1,10])
    
    DXS, DXS_intersect, DXL, DXL_intersect = get_diffusion_coefficients(stabilogram_points_ML,times,short_term_window=[0,0.5],long_term_window=[2,10])
    DTXC, X2 = get_linear_intersection(DXS, DXS_intersect, DXL, DXL_intersect)
    HXS, HXS_intersect, HXL, HXL_intersect = get_scaling_exponents(stabilogram_points_ML,times,short_term_window=[0,0.4],long_term_window=[1,10])
    
    DYS, DYS_intersect, DYL, DYL_intersect = get_diffusion_coefficients(stabilogram_points_AP,times,short_term_window=[0,0.5],long_term_window=[2,10])
    DTYC, Y2 = get_linear_intersection(DYS, DYS_intersect, DYL, DYL_intersect)
    HYS, HYS_intersect, HYL, HYL_intersect = get_scaling_exponents(stabilogram_points_AP,times,short_term_window=[0,0.4],long_term_window=[1,10])


    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times,y=stabilogram_points, mode='lines', name='Joint direction', legendgroup='Joint direction'))
    fig.add_trace(go.Scatter(x=times,y=stabilogram_points_ML, mode='lines', name='ML direction', legendgroup='ML direction'))
    fig.add_trace(go.Scatter(x=times,y=stabilogram_points_AP, mode='lines', name='AP direction', legendgroup='AP direction'))
    fig.add_trace(go.Scatter(x=[DTRC],y=[R2], mode='markers', legendgroup='Joint direction', showlegend=False, marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[DTXC],y=[X2], mode='markers', legendgroup='ML direction', showlegend=False, marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[DTYC],y=[Y2], mode='markers', legendgroup='AP direction', showlegend=False, marker=dict(size=10)))
    fig.add_shape(type="line", xref="x", yref="paper",x0=0.5, y0=0, x1=0.5,y1=1,line=dict(color="DarkOrange",width=1,),)
    fig.add_shape(type="line", xref="x", yref="paper",x0=2, y0=0, x1=2,y1=1,line=dict(color="DarkOrange",width=1,),)
#    fig.add_shape(type="line", xref="x", yref="y",x0=0, y0=DRS_intersect, x1=10,y1=10 * DRS + DRS_intersect,line=dict(color="crimson",width=1,),)
#    fig.add_shape(type="line", xref="x", yref="y",x0=0, y0=DRL_intersect, x1=10,y1=10*DRL + DRL_intersect,line=dict(color="cyan",width=1,),)
    
    fig.add_trace(go.Scatter(x=[0,10],y=[DRS_intersect,10*DRS+DRS_intersect], mode='lines', legendgroup='Joint direction', showlegend=False, opacity=0.5))
    fig.add_trace(go.Scatter(x=[0,10],y=[DRL_intersect,10*DRL+DRL_intersect], mode='lines', legendgroup='Joint direction', showlegend=False, opacity=0.5))
    
    fig.add_trace(go.Scatter(x=[0,10],y=[DXS_intersect,10*DXS+DXS_intersect], mode='lines', legendgroup='ML direction', showlegend=False, opacity=0.5))
    fig.add_trace(go.Scatter(x=[0,10],y=[DXL_intersect,10*DXL+DXL_intersect], mode='lines', legendgroup='ML direction', showlegend=False, opacity=0.5))
    
    fig.add_trace(go.Scatter(x=[0,10],y=[DYS_intersect,10*DYS+DYS_intersect], mode='lines', legendgroup='AP direction', showlegend=False, opacity=0.5))
    fig.add_trace(go.Scatter(x=[0,10],y=[DYL_intersect,10*DYL+DYL_intersect], mode='lines', legendgroup='AP direction', showlegend=False, opacity=0.5))
    fig.show()
#    fig.update_shapes(visible=False)

        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times,y=stabilogram_points, mode='lines', name='Joint direction', legendgroup='Joint direction'))
    fig.add_trace(go.Scatter(x=times,y=stabilogram_points_ML, mode='lines', name='ML direction', legendgroup='ML direction'))
    fig.add_trace(go.Scatter(x=times,y=stabilogram_points_AP, mode='lines', name='AP direction', legendgroup='AP direction'))
    fig.update_yaxes(type="log")
    fig.update_xaxes(type="log")
#    fig.update_xaxes(type="log",range=[np.log10(0.1),np.log10(12)])
#    fig.add_shape(type="line", xref="x", yref="paper",x0=0.01, y0=0, x1=0.01,y1=1,line=dict(color="DarkOrange",width=1,),)    
    fig.add_shape(type="line", xref="x", yref="paper",x0=0.4, y0=0, x1=0.4,y1=1,line=dict(color="DarkOrange",width=1,),)    
    fig.add_shape(type="line", xref="x", yref="paper",x0=1, y0=0, x1=1,y1=1,line=dict(color="DarkOrange",width=1,),)    
    
#    y = x**m * np.exp(HRS_intersect)
    
    fig.add_trace(go.Scatter(x=times,y=times**HRS*(10**HRS_intersect), mode='lines', legendgroup='Joint direction', showlegend=False, opacity=0.5))
    fig.add_trace(go.Scatter(x=times,y=times**HRL*(10**HRL_intersect), mode='lines', legendgroup='Joint direction', showlegend=False, opacity=0.5))
    
    fig.add_trace(go.Scatter(x=times,y=times**HXS*(10**HXS_intersect), mode='lines', legendgroup='ML direction', showlegend=False, opacity=0.5))
    fig.add_trace(go.Scatter(x=times,y=times**HXL*(10**HXL_intersect), mode='lines', legendgroup='ML direction', showlegend=False, opacity=0.5))
    
    fig.add_trace(go.Scatter(x=times,y=times**HYS*(10**HYS_intersect), mode='lines', legendgroup='AP direction', showlegend=False, opacity=0.5))
    fig.add_trace(go.Scatter(x=times,y=times**HYL*(10**HYL_intersect), mode='lines', legendgroup='AP direction', showlegend=False, opacity=0.5))
    
    fig.show()

    return


