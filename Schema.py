import datetime
from pydantic import BaseModel


class ValidatorMOR(BaseModel):
    nameReservoir: str
    wellNumberColumn: str
    nameDate: datetime.datetime
    workHorizon: str = None
    oilProduction: int = 0
    fluidProduction: int = 0
    waterInjection: int = 0
    timeProduction: int = 0
    timeInjection: int = 0
    coordinateXT1: int = 0
    coordinateYT1: int = 0
    coordinateXT3: int = 0
    coordinateYT3: int = 0
    workMarker: str = None
    wellStatus: str = None
