# -*- coding: utf-8 -*-
import time
from time import clock

def sleeping():
	time.sleep(5)

time.clock()
sleeping()
time.clock()
print(time.clock())
#print(str((time.clock()-a)))

b = time.time()
sleeping()
print(str(time.time()-b))


#真实时间需要用time.time() clock在Linux系统上市cpu时间，没有什么意义。