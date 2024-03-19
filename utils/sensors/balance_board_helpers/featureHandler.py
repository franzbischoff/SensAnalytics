from utils.sensors.balance_board_helpers import feature_helper as helper
import pandas as pd

def get_feature_names():
    
    area_feature_names = ['area95', 'swayarea', 'area95_majoraxis_angle', 'area95_majoraxis_length', 'area90_ML', 'area90_AP']
    visual_area_feature_names = ['RR_swayarea','VRI_swayarea']
    
    displacement_feature_names = ['pathlength', 'pathlength_ML', 'pathlength_AP', 'rms_displacement', 'rms_displacement_ML', 'rms_displacement_AP', 'std_displacement_ML', 'std_displacement_AP', 'avg_displacement', 'avg_displacement_ML', 'avg_displacement_AP', 'displacement_range_ML', 'displacement_range_AP', 'peak_displacement_ML', 'peak_displacement_AP', 'peak_displacement_forward', 'peak_displacement_backward', 'peak_displacement_left', 'peak_displacement_right', 'direction_index_ML', 'direction_index_AP', 'swayratio_ML', 'swayratio_AP', 'swaymovement_ML', 'swaymovement_AP']
    visual_displacement_feature_names = ['RR_pathlength', 'VRI_rms_displacement_ML', 'VRI_rms_displacement_AP', 'VRI_pathlength', 'VRI_avg_displacement_ML', 'VRI_avg_displacement_AP']
    
    stability_feature_names = []
    
    otherStabilometric_feature_names = ['surfacelength_ratio', 'planardeviation', 'phaseplaneparameter']
    
    velocity_feature_names = ['avg_velocity', 'avg_velocity_ML', 'avg_velocity_AP', 'peak_velocity_forward', 'peak_velocity_backward', 'peak_velocity_left', 'peak_velocity_right']
    
    bandpower_feature_names = ['bandpower_2-4_ML', 'bandpower_2-4_AP', 'bandpower_4-7_ML', 'bandpower_4-7_AP']
    
    frequencypower_feature_names = ['frequency95_ML', 'frequency95_AP', 'frequency90_ML', 'frequency90_AP', 'frequency85_ML', 'frequency85_AP', 'frequency80_ML', 'frequency80_AP', 'frequency70_ML', 'frequency70_AP']
    
    otherSpectral_feature_names = ['totalenergy_ML', 'totalenergy_AP']
    
    diffusion_feature_names = ['DTXC', 'DTYC', 'DTRC', 'X2', 'Y2', 'R2', 'DXS', 'DYS', 'DRS', 'HXS', 'HYS', 'HRS', 'DXL', 'DYL', 'DRL', 'HXL', 'HYL', 'HRL']
    other_feature_names = ['fractaldimension', 'covariance']
    
    polar_feature_names = ['swayvector_length', 'swayvector_angle', 'avg_radius']
    
    complexity_feature_names = ['sample_entropy_ML', 'sample_entropy_AP']
    
    otherNonelinear_feature_names = ['DLE_ML', 'DLE_AP']
    
    RQA_feature_names = ['%recurrence_ML', '%recurrence_AP', '%determinism_ML', '%determinism_AP', 'RQA_entropy_ML', 'RQA_entropy_AP', 'RQA_maxline_ML', 'RQA_maxline_AP', 'RQA_trend_ML', 'RQA_trend_AP']
    
    
    feature_names = area_feature_names + displacement_feature_names + stability_feature_names + otherStabilometric_feature_names + velocity_feature_names + bandpower_feature_names + frequencypower_feature_names + otherSpectral_feature_names + diffusion_feature_names + other_feature_names + polar_feature_names + complexity_feature_names + otherNonelinear_feature_names + RQA_feature_names

    visual_feature_names = visual_area_feature_names + visual_displacement_feature_names
    
    return feature_names, visual_feature_names


def get_all_features(data):

    features = {}
    
    features['area95'] = helper.get_area95(data)
    features['swayarea'] = helper.get_swayarea(data)
    features['area95_majoraxis_angle'] = helper.get_area95majoraxis(data)
    features['area95_majoraxis_length'],features['area95_minoraxis_length'] = helper.get_area95_axis_length(data)
    features['area95_majoraxis_tangent'] = helper.get_area95_minoraxis_tangent(data)
    features['markedarea'] = helper.get_markedarea(data)
    features['area90_ML'],features['area90_AP'] = helper.get_area90_length(data)
    
    features['pathlength'], features['pathlength_ML'], features['pathlength_AP'] = helper.get_pathlength(data)
    features['rms_displacement'], features['rms_displacement_ML'], features['rms_displacement_AP'] = helper.get_rms_displacement(data) 
    features['std_displacement'], features['std_displacement_ML'], features['std_displacement_AP'] = helper.get_stdev_displacement(data)
    features['avg_displacement'] = helper.get_average_displacement(data)
    features['avg_displacement_ML'], features['avg_displacement_AP'] = helper.get_average_displacement_directional(data)
    features['displacement_range_ML'], features['displacement_range_AP'] = helper.get_displacement_range(data)
    features['peak_displacement_ML'], features['peak_displacement_AP'], features['peak_displacement_forward'], features['peak_displacement_backward'], features['peak_displacement_left'], features['peak_displacement_right'] = helper.get_peak_displacements(data)
    features['direction_index_ML'], features['direction_index_AP'] = helper.get_direction_index(data)
    features['swayratio_ML'], features['swayratio_AP'] = helper.get_swayratio(data)
    features['swaymovement'], features['swaymovement_ML'], features['swaymovement_AP'] = helper.get_swaymovement(data)
    
    features['equilibriumscore'] = helper.get_equilibriumscore(data)
    
    features['surfacelength_ratio'] = helper.get_surfacelengthratio(data)
    features['planardeviation'] = helper.get_planardeviation(data)
    features['phaseplaneparameter'] = helper.get_phaseplaneparameter(data)
    
    features['avg_velocity'], features['avg_velocity_ML'], features['avg_velocity_AP'] = helper.get_average_velocity(data)
    features['peak_velocity_forward'], features['peak_velocity_backward'], features['peak_velocity_left'], features['peak_velocity_right'] = helper.get_peak_velocities(data)
    
    features['bandpower_2-4_ML'], features['bandpower_2-4_AP'] = helper.get_bandpowers(data,2,4, method=None, relative=True)
    features['bandpower_4-7_ML'], features['bandpower_4-7_AP'] = helper.get_bandpowers(data,4,7, method=None, relative=True)
    
    features['frequency95_ML'], features['frequency95_AP'] = helper.get_edgefrequency(data, 0.95, method=None)
    features['frequency90_ML'], features['frequency90_AP'] = helper.get_edgefrequency(data, 0.90, method=None)
    features['frequency85_ML'], features['frequency85_AP'] = helper.get_edgefrequency(data, 0.85, method=None)
    features['frequency80_ML'], features['frequency80_AP'] = helper.get_edgefrequency(data, 0.80, method=None)
    features['frequency70_ML'], features['frequency70_AP'] = helper.get_edgefrequency(data, 0.70, method=None)
    features['frequency95'] = helper.get_frequency95(data)
    
    features['totalenergy_ML'], features['totalenergy_AP'] = helper.get_totalenergy(data, demean=True)
    
    features['DTXC'], features['DTYC'], features['DTRC'], features['X2'], features['Y2'], features['R2'], features['DXS'], features['DYS'], features['DRS'], features['HXS'], features['HYS'], features['HRS'], features['DXL'], features['DYL'], features['DRL'], features['HXL'], features['HYL'], features['HRL'] = helper.get_diffusion_plot_analysis_features(data)
    
    features['fractaldimension'] = helper.get_fractaldimension(data)
    
    features['swayvector_length'] = helper.get_swayvectorlength(data)
    features['swayvector_angle'] = helper.get_swayvectorangle(data)
    features['avg_radius'] = helper.get_averageradius(data)
    
    features['covariance'] = helper.get_covariance(data)
    
    features['sample_entropy_ML'], features['sample_entropy_AP'] = helper.get_sampleentropy(data)
    
    features['DLE_ML'], features['DLE_AP'] = helper.get_DLE(data)
    
    features['%recurrence_ML'], features['%recurrence_AP'], features['%determinism_ML'], features['%determinism_AP'], features['RQA_entropy_ML'], features['RQA_entropy_AP'], features['RQA_maxline_ML'], features['RQA_maxline_AP'], features['RQA_trend_ML'], features['RQA_trend_AP'] = helper.get_RQA_features(data)


    return features
    
    
    
####################### EO/EO ####################
    
    
    
def get_all_visual_features(eo_data, ec_data):
    
    features = {}
    
    features['RR_swayarea']  = helper.get_swayarea_romberg(eo_data,ec_data)
    features['VRI_swayarea'] = helper.get_swayarea_vri(eo_data,ec_data)
    
    features['RR_pathlength'], features['RR_pathlength_ML'], features['RR_pathlength_AP'] = helper.get_pathlength_romberg(eo_data,ec_data)
    features['VRI_rms_displacement'], features['VRI_rms_displacement_ML'], features['VRI_rms_displacement_AP'] = helper.get_rms_displacement_vri(eo_data,ec_data)
    features['VRI_pathlength'], features['VRI_pathlength_ML'], features['VRI_pathlength_AP'] = helper.get_pathlength_vri(eo_data,ec_data)
    features['VRI_avg_displacement_ML'], features['VRI_avg_displacement_AP'] = helper.get_average_displacment_vri(eo_data,ec_data)
    
    return features
    
    
    
###########

def combine_features(participants,f,visual_f):
    participant_features = pd.DataFrame()

    for participant in participants.index:

        #create feature_df, a dataframe of all features
        EO_participant_file = participants.loc[participant,'EO sway file']
        EC_participant_file = participants.loc[participant,'EC sway file']

        if not isinstance(EO_participant_file,str) or not isinstance(EC_participant_file,str): continue #if either is not a str, skip

        EO_f = f.loc[EO_participant_file]
        EO_f = EO_f.rename(participant)
        EO_f.rename(lambda x: x + ' EO',inplace=True)

        EC_f = f.loc[EC_participant_file]
        EC_f = EC_f.rename(participant)
        EC_f.rename(lambda x: x + ' EC',inplace=True)
        feature_series = EO_f.append(EC_f).append(visual_f.loc[participant])

        #save this
        participant_features = participant_features.append(feature_series)

    participant_features = participant_features.dropna(axis=1, how='all')
    return participant_features




