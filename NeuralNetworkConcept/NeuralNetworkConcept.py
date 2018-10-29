
# coding: utf-8

# In[24]:


# step function ---> output has only 1 and 0

# this can't input array
def step_function(x):
    if x>0:
        return 1
    else:
        return 0


# In[25]:


# this can input array
import numpy as np
def step_function(x):
    y=x>0
    return y.astype(np.int)


# In[26]:


print(step_function(np.array([-1,3.5,3.0])))


# In[27]:


import numpy as np
import matplotlib.pyplot as plt
def step_function(x):
    return np.array(x>0,dtype=np.int)
x=np.arange(-5.0,5.0,0.1)
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


# In[28]:


# sigmoid function ---> output is continuous
def sigmoid_function(x):
    return 1/(1+np.exp(-x))
# exp can input array


# In[29]:


print(sigmoid_function(np.array([-1.0,1.0,2.0])))


# In[30]:


x=np.arange(-5.0,5.0,0.1)
y=sigmoid_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


# In[31]:


# ReLU function ---> x<=0,output=0 ; x>0,output=x
def ReLU_function(x):
    return np.maximum(0,x)


# In[32]:


x=np.arange(-5.0,5.0,0.1)
y=ReLU_function(x)
plt.plot(x,y)
plt.show()


# In[33]:


# multidimensional array
import numpy as np
a=np.array([1,2,3,4])
print(a)
print(np.ndim(a))
print(a.shape)
print(a.shape[0])


# In[50]:


b=np.array([[1,2],[3,4],[5,6]])
print(np.ndim(b))
print(b.shape)


# In[51]:


a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])
print(a.shape)
print(b.shape)
print(np.dot(a,b))


# In[52]:


a=np.array([[1,2,3],[4,5,6]])
b=np.array([[1,2],[3,4],[5,6]])
print(a.shape)
print(b.shape)
print(np.dot(a,b))


# In[54]:


a=np.array([[1,2],[3,4],[5,6]])
b=np.array([7,8])
print(a.shape)
print(b.shape)
print(np.dot(a,b))


# In[55]:


# Neural Network dot
x=np.array([1,2])
w=np.array([[1,3,5],[2,4,6]])
print(x.shape)
print(y.shape)
print(np.dot(x,w))

