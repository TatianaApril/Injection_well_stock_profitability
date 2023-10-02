import datetime
from typing import Optional
from pydantic import BaseModel, validator, conint, confloat
from typing import List
import pandas as pd
import numpy as np


class DictValidator_Coord(BaseModel):
    Reservoir_name: str
    Well_number: str
    Date: datetime.datetime
    Well_cluster: Optional[str] = 'Unknown'
    XT1: conint(gt=1000)
    YT1: conint(gt=1000)
    XT3: conint(gt=1000)
    YT3: conint(gt=1000)

    @validator("Date")
    @classmethod
    def validate_date(cls, value):
        if pd.isnull(value) or value.date() < datetime.date(year=1950, month=12, day=1):
            raise ValueError(f"Wrong date {value} for well")
        return value


class Validator_Coord(BaseModel):
    df_dict: List[DictValidator_Coord]


class DictValidator_inj(BaseModel):
    Well_number: str
    Date: datetime.datetime
    Status: Optional[str] = 'Unknown'
    Choke_size: conint(ge=0, le=32)
    Time_injection_1: conint(ge=0, le=744)
    Time_injection_2: conint(ge=0, le=744)
    Pkns: float
    Pkust: float
    Pwh: float
    Horizon: Optional[str] = 'Unknown'
    Pbf: float
    Pbh: float
    Pr: float
    Injection: float

    @validator("Date")
    @classmethod
    def validate_date(cls, value):
        if pd.isnull(value) or value.date() < datetime.date(year=1950, month=12, day=1):
            raise ValueError(f"Wrong date {value} for well")
        return value


class Validator_inj(BaseModel):
    df_dict: List[DictValidator_inj]


class DictValidator_prod(BaseModel):
    Well_number: str
    Date: datetime.datetime
    Status: Optional[str] = 'Unknown'
    Time_production_1: conint(ge=0, le=744)
    Time_production_2: conint(ge=0, le=744)
    Horizon: Optional[str] = 'Unknown'
    Pbh: float
    Kh: float
    Pr: float
    Rate_oil: float
    Rate_fluid: float
    Water_cut: confloat(ge=0, le=100)

    @validator("Date")
    @classmethod
    def validate_date(cls, value):
        if pd.isnull(value) or value.date() < datetime.date(year=1950, month=12, day=1):
            raise ValueError(f"Wrong date {value} for well")
        return value


class Validator_prod(BaseModel):
    df_dict: List[DictValidator_prod]


class DictValidator_HHT(BaseModel):
    Well_number: str
    HHT: conint(ge=0)
    Horizon: Optional[str] = 'Unknown'


class Validator_HHT(BaseModel):
    df_dict: List[DictValidator_HHT]


# sample of Dataframe: injCelles_horizon
sample_df_injCells_horizon = pd.DataFrame(columns=["№ добывающей", "Ячейка", "Куст добывающей",
                                                   "Объект", "Куст нагнетательной",
                                                   "Дата запуска ячейки", "Расстояние, м", "Нн, м", "Нд, м",
                                                   "Кдоб", "Кнаг", "Куч", "Квл",
                                                   "Куч*Квл", "Куч доб", "Куч Итог"])
#  sample of Dataframe: df_one_prod_well
sample_df_one_prod_well = pd.DataFrame(np.array(["Date",
                                                 'Qliq_fact, tons/day',
                                                 'Qoil_fact, tons/day',
                                                 "delta_Qliq, tons/day",
                                                 "delta_Qoil, tons/day",
                                                 "Сumulative fluid production, tons"]),
                                       columns=['Параметр'])
#  sample of Dataframe: df_one_inj_well
sample_df_one_inj_well = pd.DataFrame(np.array(["Date",
                                                'Qliq_fact, tons/day',
                                                'Qoil_fact, tons/day',
                                                "delta_Qliq, tons/day",
                                                "delta_Qoil, tons/day",
                                                "Number of working wells",
                                                "Injection, m3/day",
                                                "Current injection ratio, %",
                                                "Сumulative fluid production, tons",
                                                "Сumulative water injection, tons",
                                                "Injection ratio, %"]), columns=['Параметр'])
