# -*- coding: utf-8 -*-

canshu=9

for n in range(canshu)[1:]:
	a = (n*(n-1)/2)+1
	#print a
	for m in range(canshu-n)[1:]:
		if m ==1:
			print(a),
		else:
			print (a+n+1),
			a=a+n+1
			n=n+1
	print '\n'
	