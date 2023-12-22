""" Создание базы данных для расчета экономики на основе файлов в папке input|Экономика"""
import pandas as pd
import os
from loguru import logger
import sqlite3
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

logger.info("CHECKING FOR FILES")

dir_path = os.path.dirname(os.path.realpath(__file__))
logger.info(f"path:{dir_path}")

database_path = dir_path + "\\database"
database_content = os.listdir(path=database_path)

logger.info("check the content of input")
input_path = dir_path + "\\input"
input_content = os.listdir(path=input_path)
if 'Экономика' not in input_content:
    raise FileExistsError("no folder Экономика!")

economy_path = dir_path + "\\input\\Экономика"
economy_content = os.listdir(path=economy_path)

logger.info("check the content of Экономика")

if "НРФ.xlsb" not in os.listdir(path=economy_path):
    raise FileExistsError("НРФ.xlsb")
elif "Макра_долгосрочная.xlsx" not in os.listdir(path=economy_path):
    raise FileExistsError("Макра_долгосрочная.xlsx")
elif "Макра_оперативная_БП.xlsx" not in os.listdir(path=economy_path):
    raise FileExistsError("Макра_оперативная_БП.xlsx")
elif "Макра_оперативная_текущий_год.xlsx" not in os.listdir(path=economy_path):
    raise FileExistsError("Макра_оперативная_текущий_год.xlsx")

logger.info(f"Макра_оперативная_текущий_год.xlsx")
dict_macroeconomics = {'Обменный курс, руб/дол': 'exchange_rate',
                       'Таможенная пошлина на нефть из России, дол/тонна': 'customs_duty',
                       'Базовая ставка НДПИ на нефть, руб/т': 'base_rate_MET',
                       'Коэффициент Кман для расчета НДПИ и акциза, руб/т': 'K_man',
                       'КАБДТ с учетом НБУГ и НДФО для расчета НДПИ на нефть, руб/т': 'K_dt',
                       'Netback УУН Холмогоры - FOB Новороссийск, руб/тн (с учетом дисконта Трейдингу)': 'Netback',
                       'Urals (средний), дол/бар': 'Urals',
                       'Транспортные расходы для лицензионных участков III-IV группы НДД:'
                       ' поставка УУН Холмогоры - FOB Новороссийск, руб./т': 'cost_transportation'}

macroeconomics = pd.read_excel(economy_path + "\\Макра_оперативная_текущий_год.xlsx", nrows=15, usecols="A, B, O")
macroeconomics.columns = ["Параметр", macroeconomics.columns[1], macroeconomics.columns[2]]
macroeconomics = macroeconomics[macroeconomics["Параметр"].isin(dict_macroeconomics.keys())]
macroeconomics.replace(dict_macroeconomics, inplace=True)

logger.info(f"Макра_оперативная_БП.xlsx")
dict_business_plan = {'Доллар США': 'exchange_rate',
                      'Urals для расчета налогов': 'Urals',
                      'Нетбэк нефти для ННГ, Хантоса, СПД, Томскнефти, Мегиона,'
                      ' ГПН-Востока, Пальян, Толедо, Зап.Тарко-Салинского м/р': 'Netback',
                      'Экспортная пошлина на нефть': 'customs_duty',
                      'Коэффициент Кман для расчета НДПИ и акциза, руб./т': 'K_man',
                      'КАБДТ с учетом НБУГ для расчета НДПИ на нефть, руб./т': 'K_dt',
                      'НДПИ на нефть, руб./т': 'base_rate_MET',
                      'Транспортные расходы для м/р районов сдачи нефти (т.е. региона РФ, где расположен СИКН,'
                      ' на который сдается нефть с данного м/р) - Республики Башкортостан,'
                      ' Республики Коми, Удмуртской Республики, Пермского края, Тюменской области,'
                      ' Ненецкого автономного округа, Ханты-Мансийского автономного округа - Югры;'
                      ' Ямало-Ненецкого автономного округа (для участков недр,'
                      ' расположенных полностью или частично севернее 65 градуса северной широты,'
                      ' южнее 70 градуса северной широты и западнее 80 градуса восточной долготы'
                      ' в границах Ямало-Ненецкого автономного округа) за исключением м/р,'
                      ' приведенных ниже': 'cost_transportation',
                      'Ставка дисконтирования по Группе ГПН реальная': 'r'}

business_plan = pd.read_excel(economy_path + "\\Макра_оперативная_БП.xlsx", usecols="A, N:R", header=3)
business_plan.drop([0], inplace=True)
business_plan.columns = ["Параметр"] + list(business_plan.columns[1:])
business_plan = business_plan[business_plan["Параметр"].isin(dict_business_plan.keys())]
business_plan.replace(dict_business_plan, inplace=True)
business_plan = business_plan.fillna(method='ffill', axis=1).reset_index(drop=True)

macroeconomics = macroeconomics.merge(business_plan, left_on='Параметр', right_on='Параметр', how='outer')
macroeconomics = macroeconomics.fillna(method='bfill', axis=1)
macroeconomics.at[macroeconomics.loc[macroeconomics["Параметр"] == "r", "Параметр"].index[0], "Ед.изм."] = "Д.ед."
# macroeconomics.at[macroeconomics[macroeconomics["Параметр"] == "r"].index, 'Ед.изм.'] = "Д.ед."

logger.info(f"Макра_долгосрочная.xlsx")
dict_business_plan['Нетбэк нефти для  Хантоса, СПД, Томскнефти, Мегиона, ГПН-Востока, Пальян, Толедо'] = 'Netback'
business_plan = pd.read_excel(economy_path + "\\Макра_долгосрочная.xlsx", usecols="A, H:N", header=3)
business_plan.drop([0], inplace=True)
business_plan.columns = ["Параметр"] + list(business_plan.columns[1:])
business_plan = business_plan[business_plan["Параметр"].isin(dict_business_plan.keys())]
business_plan.replace(dict_business_plan, inplace=True)

macroeconomics = macroeconomics.merge(business_plan, left_on='Параметр', right_on='Параметр', how='outer')
macroeconomics = macroeconomics.fillna(method='ffill', axis=1)

logger.info(f"load НРФ.xlsx")
reservoirs_NDD = pd.read_excel(economy_path + "\\НРФ.xlsx", sheet_name="МР с НДД", header=None) \
    .replace(regex={'2': '', '6': '', ' ЮЛ': ''}).drop_duplicates(keep='last')

coefficients = pd.read_excel(economy_path + "\\НРФ.xlsx", sheet_name="Расчет НДПИ", header=1, nrows=41)
coefficients["Наименование участка недр/Общества"] = coefficients["Наименование участка недр/Общества"] \
    .replace(regex={'2': '', '6': '', ' ЮЛ': ''})
coefficients = coefficients[["Наименование участка недр/Общества", "Кв", "Кз", "Ккан"]]
coefficients = coefficients.groupby("Наименование участка недр/Общества").mean().reset_index()
coefficients.rename({"Наименование участка недр/Общества": "Месторождение"}, axis=1, inplace=True)

name_columns_FPA = "A:D, F, H, I:K, N:P, AM:AS, BA, BB, BD, BG, CG, DL, JW"
df_FPA = pd.read_excel(economy_path + "\\НРФ.xlsx", sheet_name="Ш-01.02.01.07-01, вер. 1.0", usecols=name_columns_FPA,
                       header=4).fillna(0)
df_FPA.drop([0, 1, 2], inplace=True)
df_FPA.drop(df_FPA.columns[[0, 2, 3]], inplace=True, axis=1)
df_FPA.reset_index(inplace=True, drop=True)

df_FPA.columns = ['Месторождение', '№скв.', '№куста', 'ДНС', 'Пласты', 'Состояние',
                  'Дебит жидк., м3/сут', 'Дебит жидк., т/сут', 'Дебит нефти, т/сут',
                  'Тариф на электроэнергию', 'УРЭ на ППД',
                  'УРЭ на подг. нефти', 'УРЭ Транспорт жидкости', 'УРЭ трансп. нефти',
                  'УРЭ трансп. подт. воды', 'УРЭ внешний транспорт нефти', 'Переменные расходы ППД',
                  'Переменные расходы по подготовке нефти',
                  'Переменные расходы по транспортировке нефти',
                  'Переменные коммерческие расходы',
                  'Удельные от нефти', 'Удельный расход ЭЭ на МП', 'Кд']

df_FPA['Кд'] = df_FPA['Кд'].replace(regex={'ТРИЗ ': '', ",": "."}).astype("float")
df_FPA['Кд'] = df_FPA['Кд'].replace(0, 1)

df_FPA["Уделка на нефть, руб/тн.н"] = df_FPA["Тариф на электроэнергию"] * (df_FPA['УРЭ на подг. нефти'] +
                                                                           df_FPA['УРЭ трансп. нефти'] +
                                                                           df_FPA['УРЭ внешний транспорт нефти']) \
                                      + df_FPA["Переменные расходы по подготовке нефти"] \
                                      + df_FPA["Переменные расходы по транспортировке нефти"] \
                                      + df_FPA["Переменные коммерческие расходы"] \
                                      + df_FPA["Удельные от нефти"]

df_FPA["Уделка на закачку, руб/м3"] = df_FPA["Тариф на электроэнергию"] * df_FPA["УРЭ на ППД"] \
                                      + df_FPA["Переменные расходы ППД"]

df_FPA["Уделка на жидкость, руб/т"] = df_FPA["Тариф на электроэнергию"] * (df_FPA["УРЭ Транспорт жидкости"]
                                                                           + df_FPA["Удельный расход ЭЭ на МП"])

df_FPA["Уделка на воду, руб/м3"] = df_FPA["Тариф на электроэнергию"] * df_FPA["УРЭ трансп. подт. воды"]

df_FPA = df_FPA[['Месторождение', '№скв.', '№куста', 'ДНС', 'Пласты', 'Состояние',
                 'Дебит жидк., м3/сут', 'Дебит жидк., т/сут', 'Дебит нефти, т/сут',
                 "Уделка на нефть, руб/тн.н", 'Уделка на закачку, руб/м3',
                 'Уделка на жидкость, руб/т', "Уделка на воду, руб/м3", 'Кд']]

df_FPA['№скв.'] = df_FPA['№скв.'].astype("str")

logger.info(f"CREATE BASE: Экономика")

connection = sqlite3.connect(database_path + f'//Экономика.db')
macroeconomics.to_sql("macroeconomics", connection, if_exists="replace", index=False)
coefficients.to_sql("coefficients", connection, if_exists="replace", index=False)
df_FPA.to_sql("df_FPA", connection, if_exists="replace", index=False)
reservoirs_NDD.to_sql("reservoirs_NDD", connection, if_exists="replace", index=False)
connection.commit()
connection.close()

logger.info("good end :)")
