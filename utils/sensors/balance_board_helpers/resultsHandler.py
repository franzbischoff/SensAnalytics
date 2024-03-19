import numpy as np
import pandas as pd
from package import dataHandler as dh
from package import featureHandler as fh

def get_effect_sizes(participants,features,visual_features=None,return_dataframe=False):
    """
    
    features: df sampled at a specific fs
    """


    PD_participants = dh.df_retrieve(participants,{'is PD': True})
    PD_sway_files = PD_participants['EC sway file'].dropna().tolist() + PD_participants['EO sway file'].dropna().tolist()

    control_participants = dh.df_retrieve(participants,{'is PD': False})
    control_sway_files = control_participants['EC sway file'].dropna().tolist() + control_participants['EO sway file'].dropna().tolist()
    
    feature_names, visual_feature_names = fh.get_feature_names()
    effect_size = pd.DataFrame(columns=feature_names)
    
    PD_indices = features.index.isin(PD_sway_files)
    control_indices = features.index.isin(control_sway_files)
    
    effect_size = _get_effect_size(effect_size,features,PD_indices,control_indices)
    
    if visual_features is not None:
        visual_effect_size = pd.DataFrame(columns=visual_feature_names)
        visual_PD_indices = visual_features.index.isin(PD_participants.index)
        visual_control_indices = visual_features.index.isin(control_participants.index)
        visual_effect_size = _get_effect_size(visual_effect_size,visual_features,visual_PD_indices,visual_control_indices)
        effect_size = effect_size.join(visual_effect_size)

    
    if return_dataframe: return effect_size
    else: return effect_size.loc['Hedges g']



def _get_effect_size(effect_size,features,PD_indices,control_indices):
    """Takes in effect_size dataframe to populate"""

    effect_size.at['PD mean'] = features.loc[PD_indices].mean(axis=0)
    effect_size.at['PD stdev'] = features.loc[PD_indices].std(axis=0) #sample stdev
    effect_size.at['PD n'] = features.loc[PD_indices].count(axis=0)

    effect_size.at['control mean'] = features.loc[control_indices].mean(axis=0)
    effect_size.at['control stdev'] = features.loc[control_indices].std(axis=0) #sample stdev
    effect_size.at['control n'] = features.loc[control_indices].count(axis=0)

    sd_pooled_numerator = (effect_size.loc['PD n']-1)*effect_size.loc['PD stdev']**2 + (effect_size.loc['control n']-1)*effect_size.loc['control stdev']**2
    sd_pooled_denominator = effect_size.loc['PD n'] + effect_size.loc['control n'] - 2
    sd_pooled = (sd_pooled_numerator/sd_pooled_denominator).pow(0.5)

    effect_size.at['Hedges g'] = (effect_size.loc['PD mean'] - effect_size.loc['control mean'])/sd_pooled
    
    return effect_size
