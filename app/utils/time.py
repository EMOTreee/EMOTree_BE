from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    KST = ZoneInfo("Asia/Seoul")
except ZoneInfoNotFoundError:
    KST = timezone(timedelta(hours=9))

def now_kst() -> datetime:
    return datetime.now(KST)
