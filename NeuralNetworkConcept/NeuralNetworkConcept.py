
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


# In[17]:


# Neural Network ---> four layers --- one input(two nodes),two middle(three nodes and two nodes),and one output(two nodes)

import numpy as np

def sigmoid_function(x):
    return 1/(1+np.exp(-x))

def lastlayer_function(x):
    return x

# input to 1st middle
X=np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])
A1=np.dot(X,W1)+B1
Z1=sigmoid_function(A1)

# 1st middle to 2st middle
W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])
A2=np.dot(Z1,W2)+B2
Z2=sigmoid_function(A2)

# 2st middle to output
W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])
A3=np.dot(Z2,W3)+B3
Y=lastlayer_function(A3)

print(Y)


# In[18]:


# Neural Network ---> four layers --- one input(two nodes),two middle(three nodes and two nodes),and one output(two nodes)
# (another)
# only W use capital letter

import numpy as np

def sigmoid_function(x):
    return 1/(1+np.exp(-x))

def lastlayer_function(x):
    return x
    
def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    
    return network

def forward(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    
    a1=np.dot(x,W1)+b1
    z1=sigmoid_function(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid_function(a2)
    a3=np.dot(z2,W3)+b3
    y=lastlayer_function(a3)
    
    return y


network=init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y)

