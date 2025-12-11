from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from sqlmodel import Session, select

from app.models.ai_monthly_report import AiMonthlyReport
from app.models.user import User
from app.routers.dependencies import get_db as get_session
from app.services.ai_report_service import generate_monthly_report

scheduler = BackgroundScheduler()

def run_monthly_report_job():
    print("📌 월간 리포트 생성 작업 시작")

    now = datetime.now()
    # 지난 달 계산
    if now.month == 1:
        last_month = 12
        last_year = now.year - 1
    else:
        last_month = now.month - 1
        last_year = now.year

    # 그 전 달
    if last_month == 1:
        prev_month = 12
        prev_year = last_year - 1
    else:
        prev_month = last_month - 1
        prev_year = last_year

    with next(get_session()) as session:
        users = session.exec(select(User)).all()

        for user in users:
            # 이미 리포트 생성되었는지 체크
            existing = session.exec(
                select(AiMonthlyReport).where(
                    AiMonthlyReport.user_id == user.id,
                    AiMonthlyReport.label_year == last_year,
                    AiMonthlyReport.label_month == prev_month
                )
            ).first()

            if existing:
                print(f"{user.id} 유저는 이미 리포트가 생성됨. 스킵")
                continue

            # 🔥 LLM을 이용해 분석 텍스트 자동 생성

            report_data = generate_monthly_report(
                session=session,
                user_id=user.id,
                prev_year=prev_year,
                prev_month=prev_month,
                year=last_year,
                month=last_month,
            )

            quiz_analysis = report_data.get("quiz_analysis")
            empathy_analysis = report_data.get("empathy_analysis")
            expression_analysis = report_data.get("expression_analysis")

            print(quiz_analysis)
            print(empathy_analysis)
            print(expression_analysis)

            # DB 저장
            report = AiMonthlyReport(
                user_id=user.id,
                label_year=prev_year,
                label_month=prev_month,
                quiz_analysis=quiz_analysis,
                empathy_analysis=empathy_analysis,
                expression_analysis=expression_analysis,
            )
            session.add(report)
            session.commit()

            print(f"✅ {user.id} 유저 리포트 생성 완료")

    print("🎉 월간 리포트 작업 종료!")

def start_scheduler():
    scheduler.add_job(
        run_monthly_report_job,
        trigger="cron",
        day=1, hour=0, minute=0,
        id="monthly_report_job",
        replace_existing=True,
    )
    scheduler.start()
    run_monthly_report_job()
