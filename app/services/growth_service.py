from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.schemas.growth_schema import (
    Emotion, Month, EmpathyCategory, MonthlyReport, ReportSchema, EmpathyTypeWithRatio
)
from sqlalchemy import func, Float
from app.models.ai_monthly_report import AiMonthlyReport
from app.models.emotion_quiz_result import EmotionQuizResult
from app.models.empathy_training_result import EmpathyTrainingResult
from app.models.emotion_expression_result import EmotionExpressionResult
from app.models.empathy_type import EmpathyType

MONTH_MAP = {
    1: Month.JAN, 2: Month.FEB, 3: Month.MAR, 4: Month.APR, 5: Month.MAY, 6: Month.JUN,
    7: Month.JUL, 8: Month.AUG, 9: Month.SEP, 10: Month.OCT, 11: Month.NOV, 12: Month.DEC
}

def get_monthly_stat(model, user_id: int, year: int, month: int, session: Session):
    start_date = datetime(year, month, 1)
    end_month = month + 1 if month < 12 else 1
    end_year = year if month < 12 else year + 1
    end_date = datetime(end_year, end_month, 1)
        
    if hasattr(model, "emotion_label"):
        emotion_field = model.emotion_label
    elif hasattr(model, "target_emotion"):
        emotion_field = model.target_emotion
    else:
        raise AttributeError(
            f"{model.__name__} 모델에 emotion_label 또는 target_emotion 컬럼이 없습니다."
        )
    
    query = (
        session.query(
            emotion_field,
            (
                func.avg(
                    model.empathy_score
                    if hasattr(model, "empathy_score")
                    else model.expression_score
                    if hasattr(model, "expression_score")
                    else func.cast(model.is_correct, Float) * 100
                )
            ).label("avg_score")
        )
        .filter(model.user_id == user_id)
        .filter(model.created_at >= start_date)
        .filter(model.created_at < end_date)
        .group_by(emotion_field)
    )

    results = {e: None for e in Emotion}
    for row in query.all():
        results[row[0]] = float(row[1]) if row[1] is not None else None
    return results

def get_growth_data(user_id: int, session: Session):
    today = datetime.today()

    interpret_growth, empathy_growth, express_growth = [], [], []

    # 지난 12개월
    for emotion in Emotion:
        interpret_list = []
        empathy_list = []
        express_list = []

        for i in range(12):
            dt = today - timedelta(days=30 * (11 - i))  # 단순히 30일 단위
            y = dt.year
            m = dt.month

            interpret_stats = get_monthly_stat(EmotionQuizResult, user_id, y, m, session)
            empathy_stats = get_monthly_stat(EmpathyTrainingResult, user_id, y, m, session)
            express_stats = get_monthly_stat(EmotionExpressionResult, user_id, y, m, session)

            interpret_list.append({"x": MONTH_MAP[m], "y": interpret_stats.get(emotion)})
            empathy_list.append({"x": MONTH_MAP[m], "y": empathy_stats.get(emotion)})
            express_list.append({"x": MONTH_MAP[m], "y": express_stats.get(emotion)})

        interpret_growth.append({"emotion": emotion, "data": interpret_list})
        empathy_growth.append({"emotion": emotion, "data": empathy_list})
        express_growth.append({"emotion": emotion, "data": express_list})

    return {
        "interpretGrowthList": interpret_growth,
        "empathyGrowthList": empathy_growth,
        "expressGrowthList": express_growth
    }

def get_last_month_report(user_id: int, session: Session):
    # DB에서 마지막 달 레포트 조회
    last_report = (
        session.query(AiMonthlyReport)
        .filter(AiMonthlyReport.user_id == user_id)
        .order_by(AiMonthlyReport.label_year.desc(), AiMonthlyReport.label_month.desc())
        .first()
    )
    if not last_report:
        return MonthlyReport(interpret="", empathy="", express="")

    return MonthlyReport(
        interpret=last_report.quiz_analysis,
        empathy=last_report.empathy_analysis,
        express=last_report.expression_analysis
    )

def get_user_empathy_type(user_id: int, session: Session):
    counts = (
        session.query(
            EmpathyType.empathy_category,
            func.count(EmpathyType.id)
        )
        .filter(EmpathyType.user_id == user_id)
        .group_by(EmpathyType.empathy_category)
        .all()
    )

    total = sum(count for _, count in counts) or 1

    ratio_map = {
        EmpathyCategory.EMOTIONAL: 0.0,
        EmpathyCategory.COGNITIVE: 0.0,
    }

    for category, count in counts:
        ratio_map[category] = count / total

    if ratio_map[EmpathyCategory.EMOTIONAL] >= ratio_map[EmpathyCategory.COGNITIVE]:
        dominant_type = EmpathyCategory.EMOTIONAL
    else:
        dominant_type = EmpathyCategory.COGNITIVE

    return {
        "type": dominant_type,
        "ratios": {
            "emotional": round(ratio_map[EmpathyCategory.EMOTIONAL], 2),
            "cognitive": round(ratio_map[EmpathyCategory.COGNITIVE], 2),
        }
    }


def get_full_report(user_id: int, session: Session):
    growth_data = get_growth_data(user_id, session)
    monthly_report = get_last_month_report(user_id, session)
    empathy_type_data = get_user_empathy_type(user_id, session)

    return ReportSchema(
        interpretGrowthList=growth_data["interpretGrowthList"],
        empathyGrowthList=growth_data["empathyGrowthList"],
        expressGrowthList=growth_data["expressGrowthList"],
        monthlyReport=monthly_report,
        empathyType=EmpathyTypeWithRatio(**empathy_type_data)
    )