import types
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas
sns.set_style("white")
from Bootstrap import mpRun 

def uncertRound(value, uncert):
	i = 0
	while uncert*(10**i) <= 1:
		i += 1
	return (round(value, i), round(uncert, i))

def plotFeat(inData, feat, cuts=None, labels=None, params={}):
	loop = False
 	if not isinstance(cuts, types.NoneType):
 		if isinstance(cuts, types.ListType):
 			loop = True
    		if not isinstance(cuts, types.ListType):
    			print "{} plots requested, but not labels passed".format(len(cuts))
    			return -1
    		elif len(cuts) != len(labels):
    			print "{} plots requested, but {} labels passed".format(len(cuts), len(labels))
    			return -1

	plt.figure(figsize=(16, 8))
	if loop:
		for i in range(len(cuts)):
			if isinstance(params, types.ListType):
				tempParams = params[i]
			else:
				tempParams = params
			sns.distplot(inData[cuts[i]][feat], label=labels[i], **tempParams)
	else:
		sns.distplot(inData[feat], **params)
	if loop:
		plt.legend(loc='best', fontsize=16)
	plt.ylabel("Density", fontsize=24, color='black')
	plt.xlabel(feat, fontsize=24, color='black')
	plt.show()

def rocPlot(inData=None, curves=None, predName='pred_class', targetName='gen_target', labels=None, aucs=None, bootstrap=False, log=False, baseline=True, params={}):
	buildCurves = True
	mulitplot = False
	if isinstance(inData, types.NoneType) and isinstance(curves, types.NoneType):
		print "Must pass either targets and preds, or curves"
		return -1
	if isinstance(curves, types.ListType) and not isinstance(labels, types.ListType):
		print "{} curves passed, but no labels".format(len(curves))
		return -1
	elif isinstance(curves, types.ListType) and isinstance(labels, types.ListType):
		if len(curves) != len(labels):
			print "{} curves passed, but {} labels".format(len(curves), len(labels))
			return -1
		elif isinstance(curves[0], roc_curve) and isinstance(labels[0], types.StringType):
			buildCurves = False
			multiplot = True
		else:
			print "Passing multiple curves requires list of SK-Learn roc_curves and list of string labels"
			return -1
	elif isinstance(curves, roc_curve):
			buildCurves = False
	else:
		print "Passing single curves requires SK-Learn roc_curve"
		return -1

	if buildCurves:
		if not isinstance(inData, pandas.core.series.Series):
			if isinstance(inData, types.list):
				if not isinstance(inData[0], pandas.core.series.Series):
					print "Building curves requires (list of) pandas.core.series.Series"
					return -1
				elif not isinstance(labels, types.ListType):
					print "{} curves passed, but no labels".format(len(inData))
					return -1
				elif isinstance(inData, types.ListType) and isinstance(labels, types.ListType):
					if len(inData) != len(labels):
						print "{} arrays passed, but {} labels".format(len(inData), len(labels))
						return -1
					elif isinstance(inData[0], pandas.core.series.Series) and isinstance(labels[0], types.StringType):
						multiplot = True
					else:
						print "Passing multiple curves requires list of SK-Learn roc_curves and list of string labels"
						return -1
				print "{} targtes passed, but {} preds".format(len(targets), len(preds))
				return -1
			else:
				print "Building curves requires (list of) pandas.core.series.Series"
				return -1

		curves = {}
		if bootstrap:
			aucArgs = []
			if multiplot:
				for i in range(len(inData)):
					aucArgs.append({'labels':inData[i][targetName], 'preds':inData[i][predName], 'name':labels[i], 'indeces':inData[i].index.tolist()})
			else:
				aucArgs.append({'labels':inData[targetName], 'preds':inData[predName], 'name':i, 'indeces':inData.index.tolist()})
			aucs = mpRun(aucArgs, rocauc)
			meanScores = {}
			for i in labels:
			    meanScores[i] = (np.mean(aucs[i]), np.std(aucs[i])/math.sqrt(len(aucs[i])))
			    print i + ' ROC AUC, Mean = {} +- {}'.format(meanScores[i][0], meanScores[i][1])
		else:
			meanScores = {}
			if multiplot:
				for i in range(len(inData)):
					meanScores[labels[i]] = roc_auc_score(inData[i][targetName].values, inData[i][predName])
					print i + ' ROC AUC: {}'.format(meanScores[i])
			else:
				meanScores = roc_auc_score(inData[targetName].values, inData[predName])
				print 'ROC AUC: {}'.format(meanScores[i])

		if multiplot:
			for i in range(len(inData)):
				curves[labels[i]] = roc_curve(inData[i][targetName].values, inData[i][predName].values)[:2]
		else:
			curves = roc_curve(inData[targetName].values, inData[predName])

	plt.figure(figsize=[8, 8])
	if multiplot:
		for i in range(len(curves)):
			if isinstance(params, types.ListType):
				tempParams = params[i]
			else:
				tempParams = params
			if buildCurves:
				plt.plot(*curves[labels[i]], label=labels[i] + r', $auc={}\pm{}$'.format(*uncertRound(*meanScores[labels[i]])), **params)
			else:
				plt.plot(*curves[i], label=labels[i], **params)
	else:
		if buildCurves:
			plt.plot(*curves, label=labels + r', $auc={}\pm{}$'.format(*uncertRound(*meanScores)), **params)
		else:
			plt.plot(*curves, label=labels, **params)
	if baseline:
		plt.plot([0, 1], [0, 1], 'k--', label='No discrimination')
	plt.xlabel('Background acceptance', fontsize=24, color='black')
	plt.ylabel('Signal acceptance', fontsize=24, color='black')
	if len(labels):
		plt.legend(loc='best', fontsize=16)
	if log:
		plt.xscale('log', nonposx='clip')
	plt.show()