import sklearn
from scipy.sparse import lil_matrix
import numpy as np
import json
import sklearn
import random
from sklearn.metrics import roc_auc_score, recall_score

class Link:
	def __init__(self):
		pass

	def format_data_for_display(self, emb_file):
		i2e = dict()
		with open(emb_file, 'r') as r:
			line = r.readline()
			# node_id = 0
			for line in r:
				embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
				node_id = int(embeds[0])
				# embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
				i2e[node_id] = embeds[1:]

		# X = []
		# Y = []
		# for (id, label) in i2l_list:
		# X.append(i2e[id])
		# Y.append(label)
		return i2e

	def getdata(self, posfile, negfile, num):
		listdata = []
		y_true = []
		tempfile1 = []
		with open(posfile, 'r') as r:
			for line in r:
				x = line.strip('\n').split()
				tempfile1.append(x)
		a = random.sample(tempfile1, num)
		for x in a:
			listdata.append([int(x[0]), int(x[1])])
			y_true.append(int(x[2]))

		tempfile2 = []
		with open(negfile, 'r') as r:
			for line in r:
				x = line.strip('\n').split()
				tempfile2.append(x)
		a = random.sample(tempfile2, num)
		for x in a:
			listdata.append([int(x[0]), int(x[1])])
			y_true.append(int(x[2]))
		y_true = np.array(y_true)
		return listdata, y_true

	def sigmoid(self, x):
		ans = 1.0 / (1.0 + np.exp(-x))
		return ans

	def calcul(self, a, b):
		return self.sigmoid(np.dot(a, b))


	def rundata(self, listdata, oremb):
		y_scores = []
		for x in listdata:
			if(x[0] not in oremb):
				print("debug0")
				print(x[0])
				break
			if(x[1] not in oremb):
				print("debug1")
				print(x[1])
				break
			y_scores.append(self.calcul(oremb[x[0]], oremb[x[1]]))
		y_scores = np.array(y_scores)
		return y_scores

	def funcauc(self, y_true, y_scores):
		ans = roc_auc_score(y_true, y_scores)
		return ans

	def funcpre(self, y_true, y_scores, num):
		the_dict = {}
		j = 0
		for i in y_scores:
			the_dict[i] = y_true[j]
			j += 1
		the_value = []
		for i in range(num // 10, num + 10, num // 10):
			y = 0
			the_pre = 0
			for x in sorted(the_dict.keys(), reverse=True):
				if the_dict[x] == 1:
					the_pre += 1
				y += 1
				if y == i:
					break
			the_value.append(the_pre / i)
		return np.mean(the_value)

	def funcval(self, y_true, y_scores, val):
		total = len(y_scores)
		right = 0
		for i in range(total):
			if(y_scores[i]>=val and y_true[i] == 1):
				right = right+1
			if(y_scores[i]<val and y_true[i] == 0):
				right = right+1
		ans = round(right/total, 6)
		return ans

	def run(self, oremb,listdata, y_true, num):
		# print("run")
		y_scores = self.rundata(listdata, oremb)
		# for i in range(len(y_scores)):
		# 	print(str(i) + " " + str(y_scores[i]) + " " + str(y_true[i]))
		ans_auc = self.funcauc(y_true, y_scores)
		ans_pre = self.funcpre(y_true, y_scores, num)
		ans_acc = self.funcval(y_true, y_scores, val=0.5)
		y_pred_binary = (y_scores > 0.5).astype(int)
		ans_recall = self.funcrecall(y_true, y_pred_binary)
		return ans_auc, ans_pre, ans_acc, ans_recall

	def funcrecall(self, y_true, y_scores):
		y_pred_binary = (y_scores > 0.5).astype(int)
		ans_recall = recall_score(y_true, y_pred_binary, average='binary')
		return ans_recall

	def test(self, dataset, method):
		emb_path = '../emb/%s/%s_%s.emb' % (dataset, dataset, method)
		pos_path = '../data/%s/pos.txt' % (dataset)
		neg_path = '../data/%s/neg.txt' % (dataset)
		oremb = self.format_data_for_display(emb_path)
		'''
		for i in range(100):
			for j in range(100):
				if(i==j):
					continue
				print(str(i) + " " + str(j) + " " + str(calcul(oremb[i], oremb[j])))
		'''
		total_auc = 0
		total_pre = 0
		total_acc = 0
		total_recall = 0
		data_dict = {'dblp': 30000, 'bitotc': 3000, 'bitalpha': 2000, 'yelp': 300000, 'amms': 1200, 'ml1m': 100000,
					 'ubuntu': 70000, 'math': 60000, 'email': 60000, 'college': 8000, 'superuser': 100000,
					 'brain': 30000, 'patent': 3000, 'school':30000, 'arxivAI':60000, 'arxivCS': 100000}
		for i in range(5):
			listdata, y_true = self.getdata(pos_path, neg_path, data_dict[dataset])  # 打算用C++写
			ans_auc, ans_pre, ans_acc, recall = self.run(oremb, listdata, y_true, data_dict[dataset])
			total_auc += ans_auc
			total_pre += ans_pre
			total_acc += ans_acc
			# total_recall += recall
		print('AUC = %.4f' % (total_auc / 5))
		print('AP = %.4f' % (total_pre / 5))
		print('ACC = %.4f' % (total_acc / 5))
		# print('Recall = %.4f' % (total_recall / 5))


if __name__ == '__main__':
	dataset = 'yelp'
	method = 'HTNE'
	test = Link()
	test.test(dataset, method)
