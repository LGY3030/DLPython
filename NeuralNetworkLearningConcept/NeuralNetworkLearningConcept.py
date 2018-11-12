
# coding: utf-8

# In[4]:


# loss function ---> mean squared error

import numpy as np
def mean_squared_error(y,t):
    return (1/2)*(np.sum((y-t)**2))

t=[0,0,1,0,0,0,0,0,0,0]
y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(mean_squared_error(np.array(y),np.array(t)))

y=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(mean_squared_error(np.array(y),np.array(t)))


# In[13]:


# loss function ---> cross entropy error

import numpy as np
def cross_entropy_error(y,t):
    # need delta because np.log(0)=-inf (delta means a very small number)
    delta=1e-7
    return -np.sum(t*np.log(y+delta))

t=[0,0,1,0,0,0,0,0,0,0]
y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(cross_entropy_error(np.array(y),np.array(t)))

y=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(cross_entropy_error(np.array(y),np.array(t)))


# In[2]:


# Little batch learning

import numpy as np
from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

train_size=x_train.shape[0]
batch_size=10
batch_mask=np.random.choice(train_size,batch_size)
x_batch=x_train[batch_mask]
t_batch=t_train[batch_mask]
print(np.random.choice(60000,10))


# In[4]:


# one or batch learning
# cross entropy error
import numpy as np
from dataset.mnist import load_mnist
def cross_entropy_error(y,t):
    if y.ndim==1:
        y=y.reshape(1,y.size)
        t=t.reshape(1,t.size)
    batch=y.shape[0]
    delta=1e-7
    return -(1/batch)*np.sum(t*np.log(y+delta))


# In[98]:


import numpy as np
t = np.array([[2],[3]])
y = np.array([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],[0.1, 0.05, 0.0, 0.7, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]])

batch_size = y.shape[0]
print(batch_size)
print(np.arange(batch_size))

print(y[np.array([0]), np.array([2])])


# In[3]:


# numerical differentation
# rounding error
import numpy as np
def numerical_diff(f,x):
    h=10e-50
    return(f(x+h)-f(x))/h
print(np.float32(1e-50))


# In[4]:


# numerical differentation
import numpy as np
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)


# In[13]:


import numpy as np
import matplotlib.pylab as plt
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)
def function1(x):
    return 0.01*x**2+0.1*x
x=np.arange(0.0,20.0,0.1)
y=function1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()
print(numerical_diff(function1,5))
print(numerical_diff(function1,10))


# In[16]:


# partial numerical differentation
def function2(x):
    return x[0]**2+x[1]**2

def function_temp1(x0):
    return x0*x0+4.0**2.0
print(numerical_diff(function_temp1,3))
def function_temp2(x1):
    return 3.0**2.0+x1*x1
print(numerical_diff(function_temp2,4))

