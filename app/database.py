import pymysql

def get_db():
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="1234",
        database="emotree",
        charset="utf8",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    db_gen = get_db()
    db = next(db_gen)
    cursor = db.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            kakao_id BIGINT NOT NULL UNIQUE,
            email VARCHAR(255),
            nickname VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    db.commit()
    db.close()
