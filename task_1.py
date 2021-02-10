

import json
import os
import zipfile

z = zipfile.ZipFile('/content/friends-data.zip', 'r')
z.extractall()

def get_subtitles_dict():
  '''
    Создаем функцию вывода субтитров в виде словаря
  '''
  top_folder = "/content/friends-data"
  subtitles_dict = dict()
  subtitles_dict[os.path.basename(top_folder)] = []
  for position, dirname in enumerate(os.listdir(top_folder)):
    subtitles_dict[os.path.basename(top_folder)].append({dirname:[]})
    for filename in os.listdir(os.path.join(top_folder, dirname)):
      subtitles_file = open(os.path.join(top_folder, dirname, filename), 'r')
      subtitles_dict[os.path.basename(top_folder)][position][dirname].append({filename:subtitles_file.read()})
    
  return subtitles_dict


dict_friends_subtitles = get_table()
json_friends_subtitles = json.dump(dict_friends_subtitles, ensure_ascii = False)

def get_subtitles_frame():
  '''
    Функция для вывода субтитров в формате датафрейма 
  '''
  top_folder = "/content/friends-data" #вводим путь до разархивированного каталока
  subtitles_array = []  # Идентифицируем список для добавление в него датафрейма 
  subtitles_dict = dict() # Идентифицируем словарь для добвления в него по ключам-названиям файлов текста файла
  for position, dirname in enumerate(os.listdir(top_folder)): #Проходимся по директориям
    for filename in os.listdir(os.path.join(top_folder, dirname)): #Проходимся по файлам
      subtitles_file = open(os.path.join(top_folder, dirname, filename), 'r') #добавляем текст файла в переменную
      subtitles_dict[filename] = [subtitles_file.read()] #добавляем текст файла в словарь
  subtitles_array.append(pd.DataFrame(subtitles_dict))
  return subtitles_array

subtitles_array = get_subtitles_frame()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df1 = subtitles_array[0] 
 
# Initialize
vectorizer = TfidfVectorizer()
vectorizer.fit(df1.iloc[0])
doc_vec = vectorizer.fit_transform(df1.iloc[0]) #Создаем Tf вектор
 
# Create dataFrame
df2 = pd.DataFrame(doc_vec.toarray().transpose(),index=vectorizer.get_feature_names()) #Создаем Tf - матрицу
 
# Change column headers
df2.columns = df1.columns

most_common_word = df2.min(axis='columns').idxmin() #находим наиболее часто встречающееся слово как с самым наименьшим значением в матрице
less_common_word = df2.max(axis='columns').idxmax() #находим наименее часто встречающееся слово как с самым наибольшим значением в матрице
words_in_all_document = df2[df2.isin([0.0])].dropna(axis=0,how='any').index #Находим слова которое есть во всех документах, как индексы с всемы положительными значениями в строке 
most_popular_hero = df2.loc[['росс','рейчел','джо','моника','чендлер','фиби']].transpose().sum().idxmin() #Находим наиболее популярного героя
