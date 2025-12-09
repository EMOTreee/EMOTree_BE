from pydantic import BaseModel, Field
from typing import Optional
from app.models.enums import EmotionLabel # enum 가져오기
#


#-------------------공감 상황 생성-----------------------

# 감정 선택(Query)
class SelectedEmotionQuery(BaseModel):
    option: Optional[EmotionLabel] = Field(
        default=EmotionLabel.RANDOM,  # 기본값 지정 가능
        description="선택 가능한 감정 옵션. 기본값은 RANDOM."
    )

# 상황 생성(Response)
class SituationCreateResponse(BaseModel):
    emotion: EmotionLabel = Field(
        description="선택된 감정"
    )
    scenario: str = Field(
        description="사용자가 선택한 감정에 맞게 생성된 시나리오 텍스트"
    )


#-------------------공감 메세지 제출------------------------------

# 공감 메시지 평가 요청(RequestBody)
class EmpathyEvaluateRequest(BaseModel):
    emotion: EmotionLabel = Field( # 선택한 감정도 저장해야하기때문에 추가
        description="사용자가 선택한 감정 (EmotionLabel Enum)"
    )
    scenario: str = Field(
        description="사용자가 선택한 감정에 맞게 생성된 시나리오 텍스트"
    )
    userMessage: str = Field(
        description="사용자가 작성한 공감 메시지"
    )

# 공감 메시지 평가 결과(Response)
class EmpathyEvaluateResponse(BaseModel):
    score: int = Field(
        ge=0, #하한
        le=100, #상한
        description="모델이 판단한 공감 점수 (0~100)"
    )
    feedback: str = Field(
        description="공감 메시지에 대한 피드백"
    )
