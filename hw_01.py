

pip install pymorphy2

import tsv
import pandas as pd
import numpy as np
import seaborn as sns
from pandas import DataFrame
import csv
import re
import pymorphy2
from pymorphy2 import tokenizers
import nltk
from collections import Counter
from nltk.corpus import stopwords
nltk.download('stopwords')

"""## Первый пункт

"""

'''
  Открываем tsv файл и добавляем его датафрейм edu_frame 
'''
with open("depression_data.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    edu_frame = pd.DataFrame(tsvreader, columns=['text',	'label',	'age'], dtype=float, )
 
edu_frame.drop(axis=0, index=0, inplace= True)#Удаляем лишнюю строку повторяющую названия столбцов

frame_without_na = edu_frame #Создаем фрейм без пустых значений, для следующих операций за исключением последней

frame_without_na.age = pd.to_numeric(frame_without_na.age) #переводим столбец возраста в числовой тип 
frame_without_na = frame_without_na.dropna() #удаление пустых значений

mean_age = frame_without_na['age'][frame_without_na.age.between(14, 50)].mean() #находим среднее значение возраста в диапазоне от 14 до 50 лет включительно

percent_sixteen = frame_without_na[frame_without_na.age.between(16,16)]['age'].count()/frame_without_na[frame_without_na.age.between(16,26)]['age'].count() #находим какую долю 16-летние авторы составляют в диапазоне от 16 до 26 лет включительно

count_28_30 = frame_without_na[(frame_without_na.age == 28)|(frame_without_na.age == 30)]['age'].count() #находим сколько в датасете текстов, авторам которых 28 или 30 лет

stats = frame_without_na[frame_without_na.age.between(18,30)].describe() #С помощью встроенной в pandas функции describe отображаем всю статистику по колонке age в диапазоне от 18 до 30 лет включительно

count_positive = edu_frame.label[edu_frame.label == "1"].count() #Находим число позитивных текстов

count_negative = edu_frame.label[edu_frame.label == "0"].count() #Находим число негативных текстов

"""## Второй пункт"""

def calc_sent(text):
  '''
  Функция подсчитывает количество знаков окончания предложения, что равно числу предложений в рамках русского литературного языка
  '''
  return text.count('.') + text.count('!') + text.count('?')

edu_frame['count sentences'] = edu_frame['text'].apply(lambda x:calc_sent(x)) #Применяем вышеописанную функцию к столбцу text

def make_tokenize(text):
  '''
  Функция токенизирует текста попутно удаляет стоп-слова
  '''
  return list(set(pymorphy2.tokenizers.simple_word_tokenize(text)) - set(stopwords.words('russian')) - set([str(i) for i in range(10)]))

edu_frame['count_tokens'] = edu_frame['text'].apply(lambda x: len(make_tokenize(x)))

'''
Подсчитвание количества токенов в тексте (без стоп-слов). Найдите среднее этих значений как по всей выборке, так и внутри каждого класса.
'''
agg_token = np.mean(edu_frame['count_tokens'])
agg_token_pos = np.mean(edu_frame.count_tokens[edu_frame.label == "1"])
agg_token_neg = np.mean(edu_frame.count_tokens[edu_frame.label == "0"])

def get_lemmas_list(text):
  '''
    Функция переводит текст в последовательность лемм.
  '''
  patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
  text = re.sub(patterns, ' ', text)
  return [morph.normal_forms(token)[0] for token in make_tokenize(text)]

def get_tags_grammems(text):
  '''
  функция переводит текст в последовательность частеречных тегов
  '''
  return [morph.parse(token)[0].tag.POS for token in get_lemmas_list(text)]

"""Дополнительно в список стоп слов мы можем добавить:


1.   Современные сленговые выражения заимствованные из английского, ввиду неопределенности их употребления и смысла



"""

'''
Визуализируем распределение количество-предложений/текст через гистограмму
'''
edu_frame.hist(column = 'count sentences', bins=100, density=True)

sns.distplot(edu_frame['count sentences'])

'''
Визуализируем распределение количество-предложений / текст внутри каждого класса через гистограмму
'''
sns.distplot(edu_frame['count sentences'][edu_frame.label == "1"], color="Red")
sns.distplot(edu_frame['count sentences'][edu_frame.label == "0"])

#Вывод и построение матриции корреляции и ее графика
print(edu_frame[['age',	'count sentences']][edu_frame.age.between(16,32)].corr()) 
sns.heatmap(edu_frame[['age',	'count sentences']][edu_frame.age.between(16,32)].corr())

"""Корреляция близка к нулю, количество предложений не зависит от возраста

## Задача 4
"""

def char_ngrams(text, n):
  '''
  Функция выводит список посимвольных н-грамм
  '''
  ngramm_array = []
  start = 0
  end = n
  while end < len(text)+1:
    ngramm_array.append(text[start:end])
    start += 1
    end += 1
  return ngramm_array
char_ngrams("уставшая мама мыла грязную раму", n =3)[:10]

def word_ngrams(text, n):
  '''
  Функция выводит список пословных н-грамм
  '''
  tokens = make_tokenize(text)
  ngramm_array = []
  start = 0
  end = n
  while end < len(tokens)+1:
    ngramm_array.append(tokens[start:end])
    start += 1
    end += 1
  return ngramm_array
word_ngrams("уставшая мама мыла грязную раму", n =3)

def word_ngrams(text, n):
  tokens = get_lemmas_list(text)
  ngramm_array = []
  start = 0
  end = n
  while end < len(tokens)+1:
    ngramm_array.append(tokens[start:end])
    start += 1
    end += 1
  return ngramm_array

def pos_ngrams(text, n):
  '''
  Функция выводит н-грамм лемм
  '''
  tokens = get_tags_grammems(text)
  ngramm_array = []
  start = 0
  end = n
  while end < len(tokens)+1:
    ngramm_array.append(tokens[start:end])
    start += 1
    end += 1
  return ngramm_array
pos_ngrams("уставшая мама мыла грязную раму", n =3)

corpus = [
    "мама мыла уставшую мыла",
    "высшая школа экономики",
    "компьютерная лингвистика",
    "осень наступила"
]

def build_pos_dict(corpus):
  '''
  Функция использует модуль collection для вывода словаря количества лемм
  '''
  return Counter(get_lemmas_list(" ".join(corpus)))

build_pos_dict(corpus)

Lemmas_pos = build_pos_dict(list(edu_frame['text'][edu_frame.label == "1"])[:200]) #Леммы и их число в первых 200-а строках позитивных текстов
Lemmas_neg = build_pos_dict(list(edu_frame['text'][edu_frame.label == "0"])[:200]) #Леммы и их число в первых 200-а строках негатвных текстов

intersection = dict(Lemmas_neg.most_common(15)).keys() & dict(Lemmas_pos.most_common(15)).keys() #Вывод пересечения между списками наиболее встречаемых лемм
