from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

# Enum 정의
class Emotion(str, Enum):
    JOY = "JOY"
    SADNESS = "SADNESS"
    ANGER = "ANGER"
    SURPRISE = "SURPRISE"
    ANXIETY = "ANXIETY"

class Month(str, Enum):
    JAN = "JAN"
    FEB = "FEB"
    MAR = "MAR"
    APR = "APR"
    MAY = "MAY"
    JUN = "JUN"
    JUL = "JUL"
    AUG = "AUG"
    SEP = "SEP"
    OCT = "OCT"
    NOV = "NOV"
    DEC = "DEC"

class EmpathyCategory(str, Enum):
    EMOTIONAL = "EMOTIONAL"
    COGNITIVE = "COGNITIVE"

class EmpathyTypeWithRatio(BaseModel):
    type: EmpathyCategory
    ratio: float 

# 데이터 모델
class GrowthData(BaseModel):
    x: Month
    y: Optional[float] = None

class GrowthListItem(BaseModel):
    emotion: Emotion
    data: List[GrowthData]

class MonthlyReport(BaseModel):
    interpret: Optional[str] = None
    empathy: Optional[str] = None
    express: Optional[str] = None

class ReportSchema(BaseModel):
    interpretGrowthList: List[GrowthListItem]
    empathyGrowthList: List[GrowthListItem]
    expressGrowthList: List[GrowthListItem]
    monthlyReport: MonthlyReport
    empathyType: EmpathyTypeWithRatio
