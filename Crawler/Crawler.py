#!/usr/bin/env python
# coding: utf-8

# In[8]:


from datetime import date,timedelta
from urllib.request import urlopen
from dateutil import rrule
import datetime
import pandas as pd
import numpy as np
import json
import time


# In[9]:


# 抓取每個月的資料
def getone(stock,month):
    url=("http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date="+month.strftime('%Y%m%d')+"&stockNo="+str(stock))
    data = json.loads(urlopen(url).read())
    return pd.DataFrame(data['data'],columns=data['fields'])


# In[12]:


# 抓取全部時間的資料 (利用getone function)
def getdata(stock,start,end):
    a=[0,0,0]
    i=0
    for x in start.split('-'):
        a[i]=int(x)
        i=i+1
    begin = date(a[0],a[1],a[2])
    b=[0,0,0]
    j=0
    for x in end.split('-'):
        b[j]=int(x)
        j=j+1
    final = date(b[0],b[1],b[2])
    
    data = pd.DataFrame()
    
    for month in rrule.rrule(rrule.MONTHLY, dtstart=begin, until=final):
        data = pd.concat([data,getone(stock,month)],ignore_index=True)
        time.sleep(9000.0/1000.0);
    
    return data


# In[13]:


crawl = getdata(2302,"2012-01-01","2012-12-31")
crawl.set_index("日期", inplace=True)
crawl.to_csv('2012.csv', encoding='utf_8_sig')


# In[ ]:




