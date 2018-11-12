import itertools
import joblib
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from utils import preprocess_sentences

if __name__ == '__main__':
	with open('resources/not_oracle_en.txt') as not_oracle_file:
		not_oracle_examples = not_oracle_file.readlines()

	with open('resources/oracle_en2.txt') as oracle_file:
		oracle_examples = oracle_file.readlines()

	not_oracle_examples = [x[2:] for x in not_oracle_examples]
	oracle_examples = [x[2:] for x in oracle_examples]

	preprocessed_oracles = preprocess_sentences(oracle_examples)
	preprocessed_not_oracles = preprocess_sentences(not_oracle_examples)

	all_data = preprocessed_oracles + preprocessed_not_oracles
	oracles_target = ['oracle'] * len(preprocessed_oracles)
	not_oracles_target = ['not_oracle'] * len(preprocessed_not_oracles)
	target_data = oracles_target + not_oracles_target

	X_train, X_test, y_train, y_test = train_test_split(all_data, target_data, test_size=0.25, random_state=0)

	glued_X_train = [' '.join(token) for token in X_train]
	glued_X_test = [' '.join(token) for token in X_test]

	vectorizer = TfidfVectorizer(min_df=0.2)
	X_train = vectorizer.fit_transform(glued_X_train)
	"""
	X_train_matrix = X_train_matrix.todense()
	X_train_lengths = [[len(t)] for t in X_train]
	X_train = np.append(X_train_matrix, X_train_lengths, 1)
	"""
	X_test = vectorizer.transform(glued_X_test)
	"""
	X_test_matrix = X_test_matrix.todense()	
	X_test_lengths = [[len(t)] for t in X_test]
	X_test = np.append(X_test_matrix, X_test_lengths, 1)
	"""

	def select_classifier(classifier, parameters, X_train, y_train, cv=6, verbose=0):
		grid_search = GridSearchCV(classifier, parameters, n_jobs=-1, cv=cv, verbose=verbose)
		grid_search.fit(X_train, y_train)

		means = grid_search.cv_results_['mean_test_score']
		stds = grid_search.cv_results_['std_test_score']
		params = np.array(grid_search.cv_results_['params'])

		results = sorted(list(zip(means, stds, params)), key=lambda result: result[0], reverse=True)[:5]

		for i, (mean, std, params) in enumerate(results[:5]):
			print("\t[%d]: %0.3f (+/-%0.03f) for %r" % (i + 1, mean, std, params))
		return [result[2] for result in results]


	def select_SVM(X_train, y_train):
		print("[SVM] Optimizing...")
		svm_parameters = [{'kernel': ['rbf'], 'gamma': (1e-3, 1e-4, 1e-5, "auto"),
						   'C': list(np.arange(1, 5000, 500))},
						  {'kernel': ['poly'], 'gamma': (1e-3, 1e-4, 1e-5, "auto"),
						   'C': list(np.arange(1, 5000, 500)),
						   'degree': list(range(1, 10, 1))},
						  {'kernel': ['linear'], 'C': list(np.arange(1, 5000, 500))}]

		selected_params = select_classifier(SVC(), svm_parameters, X_train, y_train, verbose=0)
		print("[SVM] Optimization done.")
		return selected_params


	def choose_best_svm(X_train, y_train):
		selected_svms = select_SVM(X_train, y_train)
		results = []
		print('[SVM] Repeating stratified k-fold (k=6) 30 times')
		for svm_params in selected_svms:
			clf = SVC(**svm_params)
			scores = []
			for i in range(30):
				cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=i)
				scores.append(cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=-1))
			means = np.array([score.mean() for score in scores])
			results.append((clf, str(svm_params), means))
		print('[SVM] Stratified k-fold finished')
		best_svm = sorted(results, key=lambda x: x[2].mean(), reverse=True)[1]
		print(
			'[SVM] Best mean performance: %0.3f (+/-%0.03f) with %s' % (best_svm[2].mean(), best_svm[2].std(), best_svm[1]))
		return best_svm


	def plot_confusion_matrix(cm, classes,
							  title='Confusion matrix',
							  cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, "%d (%0.3f)" % (cm[i, j], normalized_cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()


	def final_result(clf, clf_name, X_train, y_train):
		clf.fit(X_train, y_train)
		predict = clf.predict(X_test)
		report = classification_report(y_test, predict)
		conf_matrix = confusion_matrix(y_test, predict)
		acc = accuracy_score(y_test, predict)
		print("Final %s report" % clf_name)
		print(report)
		print("Acc: %0.3f" % acc)

		plot_confusion_matrix(conf_matrix, ['a', 'b'],
							  title='%s Confusion Matrix' % clf_name)

	
	if os.path.exists('best_svm.sav'):
		best_svm_model = joblib.load('best_svm.sav')
	else:
		best_svm = choose_best_svm(X_train, y_train)
		best_svm_model = best_svm[0]
		joblib.dump(best_svm_model, 'best_svm.sav')

	final_result(best_svm_model, 'SVM', X_train, y_train)