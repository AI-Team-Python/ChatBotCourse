import tensorflow as tf;  
import numpy as np;  



# tf.unpack（A, axis）是一个解包函数。A是一个需要被解包的对象，axis是一个解包方式的定义，默认是零，如果是零，返回的结果就是按行解包。如果是1，就是按列解包。
# 【注意：】tf1.0+已经将pack函数更名为stack函数,unpack函数对应更名为unstack函数.
A = [[1, 2, 3], [4, 2, 3]]  
B = tf.unstack(A, axis=1)  
c = tf.reduce_mean(A,1)

with tf.Session() as sess:  
    print (sess.run([B, c]) )

# http://blog.csdn.net/tengxing007/article/details/53940325

# http://blog.csdn.net/jerr__y/article/details/61195257

# 多层 http://blog.csdn.net/jerr__y/article/details/61195257

'''
# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
'''