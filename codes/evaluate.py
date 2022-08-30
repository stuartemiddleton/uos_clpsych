#-*- coding: utf-8 -*-
from __future__ import division
from sklearn.metrics import accuracy_score
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

f = open(sys.argv[1],"r")
lines = f.readlines()
f.close()

sorted_risk = sys.argv[2]	#yes / no

ori=[]
pre=[]
tag_dict = dict()

for i,l in enumerate(lines):
	s = l.strip().split(',')
	if i>0 and len(s) == 5:
		ori.append(s[4])
		pre.append(s[3])

		o = s[4]
		p = s[3]

		if p == o:
			if p not in tag_dict:
				tag_dict[p] = dict()
				tag_dict[p]['tp'] = 0
				tag_dict[p]['fp'] = 0
				tag_dict[p]['fn'] = 0
			tag_dict[p]['tp'] += 1
		else:
			if p not in tag_dict:
				tag_dict[p] = dict()
				tag_dict[p]['tp'] = 0
				tag_dict[p]['fp'] = 0
				tag_dict[p]['fn'] = 0
			if o not in tag_dict:
				tag_dict[o] = dict()
				tag_dict[o]['tp'] = 0
				tag_dict[o]['fp'] = 0
				tag_dict[o]['fn'] = 0
			tag_dict[p]['fn']+=1
			tag_dict[o]['fp']+=1
	
	elif i>0 and len(s) == 4:
		ori.append(s[3])
		o = s[3]

		if sorted_risk == 'yes':
			pre.append(s[2])
			p = s[2]
		else:
			pre.append(s[1])
			p = s[1]

		
		if p == o:
			if p not in tag_dict:
				tag_dict[p] = dict()
				tag_dict[p]['tp'] = 0
				tag_dict[p]['fp'] = 0
				tag_dict[p]['fn'] = 0
			tag_dict[p]['tp'] += 1
		else:
			if p not in tag_dict:
				tag_dict[p] = dict()
				tag_dict[p]['tp'] = 0
				tag_dict[p]['fp'] = 0
				tag_dict[p]['fn'] = 0
			if o not in tag_dict:
				tag_dict[o] = dict()
				tag_dict[o]['tp'] = 0
				tag_dict[o]['fp'] = 0
				tag_dict[o]['fn'] = 0
			tag_dict[p]['fn']+=1
			tag_dict[o]['fp']+=1

results=dict()
for tag in tag_dict:
	print(tag) 
	try:
		pr = tag_dict[tag]['tp']/(tag_dict[tag]['tp']+tag_dict[tag]['fp'])
	except:
		pr = 0
	try:
		rec = tag_dict[tag]['tp']/(tag_dict[tag]['tp']+tag_dict[tag]['fn'])
	except:
		rec = 0
	try:
		f1 = ((pr*rec)/(pr+rec))*2
	except:
		f1 = 0
	if tag not in results:
		results[tag]=dict()
	results[tag]['pr']=pr
	results[tag]['rec']=rec
	results[tag]['f1']=f1


if len(s) == 4:
	print('accuracy (tack b)\tP@severe\tP@moderate\tP@low\tR@severe\tR@moderate\tR@low\tF1@severe\tF1@moderate\tF1@low')

	print(
		'{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.2f}'.format(
			accuracy_score(ori, pre),
			results['Severe']['pr'],
			results['Moderate']['pr'],
			results['Low']['pr'],
			results['Severe']['rec'],
			results['Moderate']['rec'],
			results['Low']['rec'],
			results['Severe']['f1'],
			results['Moderate']['f1'],
			results['Low']['f1']
			)
		)
else:
	print('accuracy (task a)\tP@IE\tP@IS\tP@O\tR@IE\tR@IS\tR@O\tF1@IE\tF1@IS\tF1@O')

	print(
		'{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.2f}'.format(
			accuracy_score(ori, pre),
			results['IE']['pr'],
			results['IS']['pr'],
			results['0']['pr'],
			results['IE']['rec'],
			results['IS']['rec'],
			results['0']['rec'],
			results['IE']['f1'],
			results['IS']['f1'],
			results['0']['f1']
			)
		)




