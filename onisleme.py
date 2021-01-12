import numpy as np 
import pandas as pd 

data = pd.read_csv ("magaza_yorumlari.csv", encoding="utf-16")
data = data.dropna(axis = 0) # NaN değerleri kaldırma
data.head()

data ['Durum'].value_counts()

#Veri Seti Önişleme

import string
import re 
import nltk
from nltk.corpus import stopwords

noktalama = string.punctuation #noktalama işaretlerini verir 
etkisiz = stopwords.words("turkish") #etkisiz kelimeler
print(noktalama)
print(etkisiz)


for d in data['Görüş'].head():
    print(d + '\n-------------------------')
#etkisiz kelimelerin atılması
temp = ' '
for word in d.split():
    if word not in etkisiz and not word.isnumeric():
        temp += word + ' '
print(temp + '\n***************************')

#noktalama işaretlerinin temizlenmesi
for d in data['Görüş'].head():
    print(d + '\n-------------------------')
    temp = ' '
for word in d:
    if word not in noktalama:
        temp += word 
print(temp + '\n***************************')
d = temp

data.to_csv(r'./cleaned.csv', index = False,)