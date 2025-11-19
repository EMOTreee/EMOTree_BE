import pymysql

db = pymysql.connect(
    host="localhost",
    user="root",
    password="1234",
    database="emotree",
    charset="utf8",
    autocommit=True
)

def init_db():
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

cursor = db.cursor()
