import pandas as pd
raw_data = pd.read_csv('../data/UCI_phishing_web_duplicate.txt', header=None)
data = raw_data.values

features = data[::, :-1]
labels = data[::, -1]
for feature_list in features:
	for index_f,feature_value in enumerate(feature_list):
		if feature_value == 0:
			feature_list[index_f] = 2
		elif feature_value == -1:
			feature_list[index_f] = 3
for index_l,label_value in enumerate(labels):
	if label_value == -1:
		labels[index_l] = 2
		
print(features)
print(labels)


for index,feature_list in enumerate(features):
	for feature_value in feature_list:
		with open("UCI_phishing_web_duplicate_no_negative.txt","a") as f:
			f.write(str(feature_value)+',')
	with open("UCI_phishing_web_duplicate_no_negative.txt", "a") as f:
		f.write(str(labels[index]) + '\n')