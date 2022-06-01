from module.db import db_manger
from module.config import *
import pymysql

# db_con = pymysql.connect(
#                 host=DB_HOST,
#                 user=DB_USER,
#                 passwd=DB_PASSWORD,
#                 db=DB_NAME,
#                 charset='utf-8'
#             )
a = db_manger().table_create()