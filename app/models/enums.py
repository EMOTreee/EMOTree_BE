from enum import Enum

class EmotionLabel(str, Enum):
    RANDOM = "RANDOM" # 랜덤도 추가가 필요할 것 같아서 추가하였습니다.
    JOY = "JOY"
    ANGER = "ANGER"
    SADNESS = "SADNESS"
    SURPRISE = "SURPRISE"
    ANXIETY = "ANXIETY"
    NEUTRAL = "NEUTRAL"