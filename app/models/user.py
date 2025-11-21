from app.database import get_db

def upsert_user(kakao_id: int, email: str, nickname: str):
    db_gen = get_db()
    db = next(db_gen)
    try:
        cursor = db.cursor()
        sql = """
            INSERT INTO users (kakao_id, email, nickname)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE email=%s, nickname=%s
        """
        cursor.execute(sql, (kakao_id, email, nickname, email, nickname))
    finally:
        db.close()
        
