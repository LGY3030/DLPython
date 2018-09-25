
# coding: utf-8

# In[4]:


import numpy as np
a=np.array([1.0,2.0,6.0])
print(a)
print(type(a))


# In[6]:


x=np.array([1.0,2.0,3.0])
y=np.array([2.0,4.0,6.0])
print(x+y)
print(x-y)
print(x*y)
print(x/y)


# In[7]:


print(2*x)


# In[12]:


a=np.array([[1.0,2.0],[2.0,4.0]])
print(a)
print(a.shape)
print(a.dtype)


# In[13]:


x=np.array([[1,2],[2,4]])
y=np.array([[3,6],[4,8]])
print(x+y)
print(2*x)


# In[18]:


#broadcast
#比較 2 array的shape,只有 1.当前维度的值相等 2.当前维度的值有一个是1 可以
#2*[[1,2],[2,4]] = [[2,2],[2,2]]*[[1,2],[2,4]] 
x=np.array([[1,2],[2,4]])
y=np.array([2,4])
print(x+y)


# In[19]:


x=np.array([2,4,6,8])
y=np.array([2])
print(x+y)


# In[23]:


x=np.array([[1,2],[2,4],[3,6]])
print(x)
print(x[0])
print(x[0][1])


# In[22]:


for row in x:
    print(row)


# In[24]:


X=x.flatten()
print(X)


# In[31]:


#可用array存取元素
print(X[[0,2,4]])
print(X[np.array([0,2,4])])


# In[32]:


print(X>2)


# In[33]:


print(X[X>2])

