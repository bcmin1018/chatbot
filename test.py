from code.train_tools.db import db_manger

# db_con = pymysql.connect(
#                 host=DB_HOST,
#                 user=DB_USER,
#                 passwd=DB_PASSWORD,
#                 db=DB_NAME,
#                 charset='utf-8'
#             )
a = db_manger().db_create()