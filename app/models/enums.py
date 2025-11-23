from enum import Enum

class EmotionLabel(str, Enum):
    HAPPY = "HAPPY"
    SAD = "SAD"
    ANGRY = "ANGRY"
    SURPRISE = "SURPRISE"
    ANXIETY = "ANXIETY"