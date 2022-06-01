import pymysql
from module.config import *

class db_manger:
    def __init__(self):
        self.db_con = self.db_connect()
        # self.db_con = pymysql.connect(
    #         host   = DB_HOST,
    #         user   = DB_USER,
    #         passwd = DB_PASSWORD,
    #         db     = DB_NAME,
    #         charset= 'utf-8'
    #     )
    def db_connect(self):
        try:
            db_con = pymysql.connect(
                host   = DB_HOST,
                user   = DB_USER,
                passwd = DB_PASSWORD,
                db     = DB_NAME,
                charset= 'utf8'
            )
        except Exception as e:
            print(e)

        return db_con

    def table_create(self):
        sql = '''
            CREATE TABLE IF NOT EXISTS `chatbot_train_data` (
            `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
            `intent` VARCHAR(45) NULL,
            `ner` VARCHAR(1024) NULL,
            `query` TEXT NULL,
            `answer` TEXT NOT NULL,
            `answer_image` VARCHAR(2048) NULL,
            PRIMARY KEY (`id`))
        ENGINE = InnoDB DEFAULT CHARSET=utf8
        '''
        with self.db_con.cursor() as cursor:
            cursor.execute(sql)