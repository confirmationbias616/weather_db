import pandas as pd

def analyze():
	hpr = pd.read_csv('/Users/Alex/Dropbox (Personal)/HPResults.csv')

	hpr['date'] = hpr.log_time.apply(lambda x: x[:10])
	hpr['goal'] = (hpr.ML_rms - hpr.Mean_rms) / hpr.Mean_rms
	hpr['goal_TWN'] = (hpr.ML_rms - hpr.TWN_rms) / hpr.TWN_rms
	hpr['goal_EC'] = (hpr.ML_rms - hpr.EC_rms) / hpr.EC_rms

	starting_index = hpr[(hpr.date=="2018-11-19")&(hpr.max_features != 'auto')].index[0]

	hpr = (hpr[hpr.index>starting_index])
	hpr['max_features'] = hpr.max_features.apply(float)

	feat_analysis = hpr.corr().goal.apply(abs).drop(list(hpr.filter(regex="_rms"))).drop(list(hpr.filter(regex="_ave"))).drop('goal').dropna().sort_values(ascending=False)

	meta_hpr = hpr[['log_time']+['goal']+list(hpr.filter(feat_analysis.index))]

	meta_hpr.to_csv('/Users/Alex/Dropbox (Personal)/HPMetaResults.csv')