
# coding: utf-8

# In[5]:


def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tmp=x1*w1+x2*w2
    if tmp>theta:
        return 1
    else:
        return 0


# In[6]:


print(AND(1,1))
print(AND(0,1))
print(AND(1,0))
print(AND(0,0))


# In[11]:


#bias
import numpy as np
x=np.array([0,1])
w=np.array([0.5,0.5])
b=-0.7
print(w*x)
print(np.sum(w*x))
print(np.sum(w*x)+b)


# In[12]:


def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(w*x)+b
    if tmp>0:
        return 1
    else:
        return 0


# In[13]:


print(AND(1,1))
print(AND(0,1))
print(AND(1,0))
print(AND(0,0))


# In[14]:


def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(w*x)+b
    if tmp>0:
        return 1
    else: 
        return 0


# In[15]:


print(NAND(1,1))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(0,0))


# In[16]:


def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sum(w*x)+b
    if tmp>0:
        return 1
    else:
        return 0


# In[17]:


print(OR(1,1))
print(OR(0,1))
print(OR(1,0))
print(OR(0,0))


# In[18]:


# perceptron ---> linear
# XOR ---> nonlinear
# multiperceptron ---> XOR
def XOR(x1,x2):
    a=NAND(x1,x2)
    b=OR(x1,x2)
    return AND(a,b)


# In[19]:


print(XOR(1,1))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(0,0))

