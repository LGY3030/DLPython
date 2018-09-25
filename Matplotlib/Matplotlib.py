
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


a=np.arange(0,6,0.1)
b=np.sin(a)
plt.plot(a,b)
plt.show()


# In[11]:


a=np.arange(0,6,0.1)
b=np.sin(a)
c=np.cos(a)
plt.plot(a,b,label='sin')
plt.plot(a,c,linestyle="--",label='cos')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin&cos')
plt.legend()
plt.show()


# In[53]:


import matplotlib.pyplot as plt
from matplotlib.image import imread
img=imread('image/mountain.jpg')
plt.imshow(img)
plt.show()

