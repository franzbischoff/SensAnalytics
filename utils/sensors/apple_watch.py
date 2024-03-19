
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def create_df(a, b, x, y):
    return pd.DataFrame(np.array([a, b]).T, columns=[x, y])

def get_data(FILE):
    #FILE = '/Users/chirathhettiarachchi/Documents/PhD/RA_OHIOH/app/apple_watch/sample_export.xml'
    bpm, hr_time = [], []
    walk_double_support, walk_double_support_date = [], [] 
    running_vertical_oscillation, running_vertical_oscillation_date = [], [] 
    running_ground_contact, running_ground_contact_date = [], [] 
    apple_walking_steadiness, apple_walking_steadiness_date = [], [] 
    walking_assymetry, walking_assymetry_date = [], [] 

    root = ET.parse(FILE).getroot()
    for tag in root.findall('Record'):
        if tag.get('type') == 'HKQuantityTypeIdentifierOxygenSaturation':
            o2_sat = tag.get('value')
            for t in tag.findall('MetadataEntry'):
                baro_pressure = t.get('value')
        
        if tag.get('type') == 'HKQuantityTypeIdentifierEnvironmentalSoundReduction':
            sound = tag.get('value')
            sound_unit = tag.get('unit')
                
        if tag.get('type') == 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN':
            for t in tag.findall('HeartRateVariabilityMetadataList/InstantaneousBeatsPerMinute'):
                beats = t.get('bpm')
                time = t.get('time')
                bpm.append(beats)
                hr_time.append(time) 

        if tag.get('type') == 'HKQuantityTypeIdentifierWalkingDoubleSupportPercentage':
            walk_double_support.append(tag.get('value'))
            walk_double_support_date.append(tag.get('creationDate'))
        
        if tag.get('type') == 'HKQuantityTypeIdentifierRunningVerticalOscillation':
            running_vertical_oscillation.append(tag.get('value'))
            running_vertical_oscillation_date.append(tag.get('creationDate'))

        if tag.get('type') == 'HKQuantityTypeIdentifierRunningGroundContactTime':
            running_ground_contact.append(tag.get('value'))
            running_ground_contact_date.append(tag.get('creationDate'))
        
        if tag.get('type') == 'HKQuantityTypeIdentifierAppleWalkingSteadiness':
            apple_walking_steadiness.append(tag.get('value'))
            apple_walking_steadiness_date.append(tag.get('creationDate'))

        if tag.get('type') == 'HKQuantityTypeIdentifierWalkingAsymmetryPercentage':
            walking_assymetry.append(tag.get('value'))
            walking_assymetry_date.append(tag.get('creationDate'))
    
    apple_hr = create_df(bpm, hr_time, 'BPM', 'Time')
    double_support = create_df(walk_double_support, walk_double_support_date, 'Walk Double Support', 'Time')
    vertical_oscillation = create_df(running_vertical_oscillation, running_vertical_oscillation_date, 'Running Vertical Oscillation', 'Time')
    ground_contact = create_df(running_ground_contact, running_ground_contact_date, 'Running Ground Contact Time', 'Time')
    walk_steadiness = create_df(apple_walking_steadiness, apple_walking_steadiness_date, 'Walking Steadiness', 'Time')
    walk_assymetry = create_df(walking_assymetry, walking_assymetry_date, 'Walk Assymetry', 'Time' )
    
    apple_data_extract = {'apple_hr': apple_hr, 'o2_sat': o2_sat, 'baro_pressure': baro_pressure, 'sound': sound,
                            'sound_unit': sound_unit, 'double_support': double_support, 'vertical_oscillation': vertical_oscillation,
                            'ground_contact': ground_contact, 'walk_steadiness': walk_steadiness, 'walk_assymetry': walk_assymetry}
    
    
    return apple_data_extract