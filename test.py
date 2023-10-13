import os
import pandas as pd
import numpy as np
import yaml

df = pd.read_excel("input/Экономика/Поиск аналога.xlsx", sheet_name="СПИСОК",
                   usecols='C:D')
df.columns = ['Qliq_min',	'groupe']
dict_df = df.to_dict()

with open(r'conf_files/liquid_groups.yml', 'w') as file:
    documents = yaml.dump(dict_df, file, allow_unicode=True, encoding='utf8')

print(1)


list = [5,6,7]
list.remove(5)
# for Injection_ratio, %
volume_factor = 1.01  # volume factor of injected fluid
Rw = 1  # density of injected water g/cm3


def loader():
    """Reading data from keys"""
    with open("keys.txt", "r") as f:
        keys = eval(f.read())

    dictex = {}
    for key in keys:
        dictex[key] = pd.read_csv("data_{}.csv".format(str(key)), index_col=False)

    return dictex


dict_reservoir_df = loader()

df_forecasts = pd.DataFrame()

for key in list(dict_reservoir_df.keys()):
    if "Прогноз" in key:
        if df_forecasts.empty:
            df_forecasts = dict_reservoir_df.pop(key)
        else:
            df_forecasts = df_forecasts.append(dict_reservoir_df.pop(key), ignore_index=True)

del df_forecasts["Маркер"]
del df_forecasts["Объект"]
dict_agg = {df_forecasts.columns[2]: 'max'}
dict_agg_2 = dict.fromkeys(df_forecasts.columns[4:], 'sum')
# result = df_final_inj_well.groupby(['Ячейка', 'Параметр'])[df_final_inj_well.columns[3:]].sum().reset_index()
result = df_forecasts.groupby(['Ячейка', 'Параметр']).agg({**dict_agg, **dict_agg_2})
result.reset_index(inplace=True)




# result.insert(1, "тек. Комп на посл. месяц, %", result.iloc[7, -1])
# result.insert(1, "накоп. Комп на посл. месяц, %", result.iloc[10, -1])

data_file = "files/Максимальное расстояние реагирования для объекта .xlsx"
with open('conf_files/max_reaction_distance.yml', 'rt', encoding='utf8') as yml:
    max_reaction_distance = yaml.load(yml, Loader=yaml.Loader)

df = pd.read_excel(os.path.join(os.path.dirname(__file__), data_file), header=1, usecols='B:E')
dict_df = (df.groupby('Месторождение')
           .apply(lambda x: dict(zip(x['Объект'], df[['L реагирования, м', 'Проницаемость, мД']].values.tolist())))
           .to_dict())

with open(r'max_reaction_distance.yml', 'w') as file:
    documents = yaml.dump(dict_df, file, allow_unicode=True, encoding='utf8')

print(1)
