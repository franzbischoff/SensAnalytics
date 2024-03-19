from package import dataHandler as dh
from package import featureHandler as fh
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
import itertools

def get_participant_matches(participants):

    matches = get_fixed_participant_matches() + get_variable_participant_matches(participants,age_range=5)
    return matches


def get_fixed_participant_matches():
    """There are 10 control females and 7 viable PD females. Thus to use most of the data set possible, the 3 of the control females will need to match with 3 PD males
    
    9 of these control females will be a fixed match. 7 control and PD female matches were done so simply by ordering age by ascending order and matching them. The other 3 control females were all older, so this will create matches with the least age difference
    
    Of the 3 older control females, 2 of them will be fix matched with 2 PD males of the closest age. These PD males are not similar in age to any control males, so they would not have been utilised anyway"""
    
    female_matches = [('C010', 'P019'), #53 and 53
                      ('C031', 'P038'), #67 and 57
                      ('C030', 'P021'), #67 and 58
                      ('C028', 'P001'), #69 and 58
                      ('C024', 'P026'), #71 and 62
                      ('C025', 'P027'), #72 and 67
                      ('C014', 'P008')] #74 and 69
    mixed_matches = [('C021', 'P002'), #81 and 82
                     ('C032', 'P012')] #94 and 91
                    
    return female_matches + mixed_matches
    
    
    
    
def get_variable_participant_matches(participants, age_range=5):

    controls_to_match = participants.loc[['C004','C013','C009','C020','C006','C026']] #C026 is female, everyone else male
    viable_matches = dh.df_retrieve(participants,{'is PD': True,'Sex':'Male'})
    viable_matches = viable_matches.loc[~viable_matches.index.isin(['P002','P012','P013','P014'])] #exclude these because P002 and P012 matched already with other females, and P013 has weird CoP that results in some features being NaN
    
    #Pair controls with their potential matches
    potential_matches_df = pd.DataFrame(columns=['Possible PD matches','How many'])
    for control in controls_to_match.index:
        age = controls_to_match.loc[control,'Age']
        potential_matches = []
        for r in range(-age_range,age_range+1):
            m = dh.df_retrieve(viable_matches,{'Age':age+r})
            potential_matches += m.index.tolist()
        potential_matches_df.loc[control,'Possible PD matches'] = potential_matches
        potential_matches_df.loc[control,'How many'] = len(potential_matches)
    potential_matches_df = potential_matches_df.sort_values(by='How many')

    #def helper to remove all occurances of a person in potential matches
    def remove_match_from_potentials(df,match):
        for participant in df.index:
            possible_matches = df.loc[participant,'Possible PD matches']
            if match in possible_matches: possible_matches.remove(match)
            df.loc[participant,'Possible PD matches'] = possible_matches
        return df
                
    matches = []
    for control in potential_matches_df.index:
        possible_matches = potential_matches_df.loc[control,'Possible PD matches']
        match = random.choice(possible_matches)
        matches.append((control,match))
        
        potential_matches_df = remove_match_from_potentials(potential_matches_df,match) #remove match from all other possible choices to stop double dipping

    return matches
    

def get_ensemble_model(seed):


    clf1 = LogisticRegression(tol=1e-6, solver='liblinear', max_iter=1000, random_state=seed)
    clf2 = SVC(kernel='rbf', probability=True, tol=1e-4, max_iter=-1, random_state=seed)
    clf3 = RandomForestClassifier(random_state=seed)
    #clf4 = GaussianNB()
    #clf5 = KNeighborsClassifier(n_neighbors=3)
    #clf6 = DecisionTreeClassifier(random_state=seed)
    
#    eclf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('rf', clf3), ('gb', clf4), ('knn', clf5), ('tree', clf6)], voting='soft', n_jobs=-1)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('rf', clf3)], voting='soft', n_jobs=-1)


    return eclf

    
def get_X_y(matches, train_index, test_index, features):
    
    train_participants = list(itertools.chain(*[matches[i] for i in train_index]))
    train_features = features.loc[train_participants]
    train_X = train_features.to_numpy()
    train_y = np.array([1 if 'P' in file else 0 for file in train_features.index])
    
    test_participants = list(itertools.chain(*[matches[i] for i in test_index]))
    test_features = features.loc[test_participants]
    test_X = test_features.to_numpy()
    test_y = np.array([1 if 'P' in file else 0 for file in test_features.index])
    
    return train_X, train_y, test_X, test_y        


       

def get_all_AUROCs_with_feature_selection(participants,features,feature_selector=None, repetitions=10,return_dataframe=False,seeds=None):
    """The concept here is a little different from get_individual_AUROCs. 
    
    The point here is to use all features, that includes the EO and EC and the features that use the relationship between EO and EC. For this reason, we redefine the task as a classification of participant, rather than classification of a recording. 
    
    A total of 30 participants are used here. Each participant will have the entire list of features generated from both their sway files (EO and EC). Some redundancies will be removed (i.e., the features that are exactly the same)
    """
    
    
    scaler = preprocessing.StandardScaler()
    AUROCs = pd.DataFrame()
    for rep in range(repetitions):
        if seeds is not None:
            seed = seeds[rep]
            random.seed(seed)
            np.random.seed(seed)
        else: seed = 0

        matches = get_participant_matches(participants) 
        kf = KFold(n_splits=5,shuffle=True)
        kf.get_n_splits(matches)
        
        for fold, (train_index, test_index) in enumerate(kf.split(matches)):

            train_X, train_y, test_X, test_y=get_X_y(matches,train_index,test_index,features)
            
            train_X = scaler.fit_transform(train_X)
            test_X = scaler.transform(test_X)
            
            if feature_selector is not None:
                feature_selector.fit(train_X,train_y)
                train_X = feature_selector.transform(train_X)
                test_X = feature_selector.transform(test_X)

            eclf = get_ensemble_model(seed)
            #eclf = LogisticRegression(tol=1e-6, solver='liblinear', max_iter=1000, random_state=seed)
            eclf.fit(train_X, train_y)


            AUROC = roc_auc_score(test_y, eclf.predict_proba(test_X)[:,1])

            AUROCs.at[rep,fold] = AUROC
    
    if return_dataframe: return AUROCs
    else: return AUROCs.mean(axis=1)
    


def get_effectiveness(participants,features,feature_selector=None, repetitions=10,return_dataframe=False,seeds=None):
    """The concept here is a little different from get_individual_AUROCs. 
    
    The point here is to use all features, that includes the EO and EC and the features that use the relationship between EO and EC. For this reason, we redefine the task as a classification of participant, rather than classification of a recording. 
    
    A total of 30 participants are used here. Each participant will have the entire list of features generated from both their sway files (EO and EC). Some redundancies will be removed (i.e., the features that are exactly the same)
    """
    
    
    scaler = preprocessing.StandardScaler()
    AUROCs = pd.DataFrame()
    ACCs = pd.DataFrame()
    for rep in range(repetitions):
        if seeds is not None:
            seed = seeds[rep]
            random.seed(seed)
            np.random.seed(seed)
        else: seed = 0

        matches = get_participant_matches(participants) 
        kf = KFold(n_splits=5,shuffle=True)
        kf.get_n_splits(matches)

        for fold, (train_index, test_index) in enumerate(kf.split(matches)):

            train_X, train_y, test_X, test_y=get_X_y(matches,train_index,test_index,features)
            
            train_X = scaler.fit_transform(train_X)
            test_X = scaler.transform(test_X)
            
            if feature_selector is not None:
                feature_selector.fit(train_X,train_y)
                train_X = feature_selector.transform(train_X)
                test_X = feature_selector.transform(test_X)

            eclf = get_ensemble_model(seed)
            #eclf = LogisticRegression(tol=1e-6, solver='liblinear', max_iter=1000, random_state=seed)
            eclf.fit(train_X, train_y)


            AUROC = roc_auc_score(test_y, eclf.predict_proba(test_X)[:,1])
            acc = accuracy_score(test_y, eclf.predict(test_X))

            AUROCs.at[rep,fold] = AUROC
            ACCs.at[rep,fold] = acc
    
    if return_dataframe: return AUROCs
    else: return AUROCs.mean(axis=1)
    


def get_individual_AUROCs(participants,features,repetitions=10,return_dataframe=False,seeds=None):
    """
    features: df sampled at a specific fs
    """
    folds = 5
    
    AUROCs = pd.DataFrame()
    scaler = preprocessing.StandardScaler()


    for rep in range(repetitions):
        if seeds is not None:
            seed = seeds[rep]
            random.seed(seed)
            np.random.seed(seed)
        else: seed = 0

        matches = get_participant_matches(participants)
        kf = KFold(n_splits=folds,shuffle=True)
        kf.get_n_splits(matches)
        
        kf_AUROCs = pd.DataFrame()
        for fold, (train_index, test_index) in enumerate(kf.split(matches)):
            
            for feature_name in features.columns:
                
                train_X, train_y, test_X, test_y = get_X_y(matches,train_index,test_index,features[[feature_name]])
                train_X = scaler.fit_transform(train_X)
                test_X = scaler.transform(test_X)
                
                eclf = get_ensemble_model(seed)
#                eclf = LogisticRegression(tol=1e-6, solver='liblinear', max_iter=1000, random_state=seed)
                eclf.fit(train_X, train_y)
                
                AUROC = roc_auc_score(test_y, eclf.predict_proba(test_X)[:,1])
                kf_AUROCs.at[feature_name,fold] = AUROC
        
        AUROCs[rep] = kf_AUROCs.mean(axis=1)
                
    if return_dataframe: return AUROCs
    else: return AUROCs.mean(axis=1)
    
    
    
def get_individual_effectiveness(participants,features,repetitions=10,return_dataframe=False,seeds=None):
    """
    features: df sampled at a specific fs
    """
    folds = 5
    
    AUROCs = pd.DataFrame()
    ACCs = pd.DataFrame()
    scaler = preprocessing.StandardScaler()


    for rep in range(repetitions):
        if seeds is not None:
            seed = seeds[rep]
            random.seed(seed)
            np.random.seed(seed)
        else: seed = 0

        matches = get_participant_matches(participants)
        kf = KFold(n_splits=folds,shuffle=True)
        kf.get_n_splits(matches)
        
        kf_AUROCs = pd.DataFrame()
        kf_ACCs = pd.DataFrame()
        for fold, (train_index, test_index) in enumerate(kf.split(matches)):
            
            for feature_name in features.columns:
                
                train_X, train_y, test_X, test_y = get_X_y(matches,train_index,test_index,features[[feature_name]])
                train_X = scaler.fit_transform(train_X)
                test_X = scaler.transform(test_X)
                
                eclf = get_ensemble_model(seed)
                #eclf = LogisticRegression(tol=1e-6, solver='liblinear', max_iter=1000, random_state=seed)
                eclf.fit(train_X, train_y)
                
                AUROC = roc_auc_score(test_y, eclf.predict_proba(test_X)[:,1])
                acc = accuracy_score(test_y, eclf.predict(test_X))
                
                kf_AUROCs.at[feature_name,fold] = AUROC
                kf_ACCs.at[feature_name,fold] = acc
        
        AUROCs[rep] = kf_AUROCs.mean(axis=1)
        ACCs[rep] = kf_ACCs.mean(axis=1)
                
    if return_dataframe: return AUROCs,ACCs
    else: return AUROCs.mean(axis=1),ACCs.mean(axis=1)
    
    



