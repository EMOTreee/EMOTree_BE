from app.database import cursor

def upsert_user(kakao_id: int, email: str, nickname: str):
    sql = """
        INSERT INTO users (kakao_id, email, nickname)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE email=%s, nickname=%s
    """
    cursor.execute(sql, (kakao_id, email, nickname, email, nickname))
