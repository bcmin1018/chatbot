import pymysql
from config.DatabaseConfig import *

db = None
try:
    db_con = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        passwd=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8'
    )

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
    with db_con.cursor() as cursor:
        cursor.execute(sql)

except Exception as e:
    print(e)

finally:
    if db_con is not None:
        db_con.close()

# class db_manger:
#     def __init__(self):
#         self.db_con = self.db_connect()
#
#     def db_connect(self):
#         try:
#             db_con = pymysql.connect(
#                 host   = DB_HOST,
#                 user   = DB_USER,
#                 passwd = DB_PASSWORD,
#                 db     = DB_NAME,
#                 charset= 'utf8'
#             )
#         except Exception as e:
#             print(e)
#
#         return db_con
#
#     def query(self, sql):
#         with self.db_con.cursor() as cursor:
#             cursor.execute(sql)
#
#     def db_create(self):
#         sql = '''
#         CREATE DATABASE chatbot
#         '''
#         self.query(sql)
#
#     def table_create(self):
#         sql = '''
#             CREATE TABLE IF NOT EXISTS `chatbot_train_data` (
#             `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
#             `intent` VARCHAR(45) NULL,
#             `ner` VARCHAR(1024) NULL,
#             `query` TEXT NULL,
#             `answer` TEXT NOT NULL,
#             `answer_image` VARCHAR(2048) NULL,
#             PRIMARY KEY (`id`))
#         ENGINE = InnoDB DEFAULT CHARSET=utf8
#         '''
#         with self.db_con.cursor() as cursor:
#             cursor.execute(sql)