import urllib

from dotenv import dotenv_values
from sqlalchemy import create_engine


class SQLConnection:

    def __init__(self, db="db4free"):
        config = dotenv_values(f"src\\{db}.env")

        conn = f"mysql+pymysql://{config['Uid']}:{config['Pwd']}@{config['Server']}/{config['Database']}?charset=utf8"

        # quoted = urllib.parse.quote_plus(conn)

        self.engine = create_engine(conn, echo=False)

    # def __init__(self, db='db4free') -> None:
    #     config = dotenv_values(f'src\\{db}.env')
    #     conn = ('Driver={ODBC Driver 17 for SQL Server};'
    #             f'Server={config["Server"]},3306;'
    #             f'Database={config["Database"]};'
    #             f'Uid={config["Uid"]};'
    #             f'Pwd={config["Pwd"]};'
    #             'Encrypt=yes;'
    #             'TrustServerCertificate=no;'
    #             'Connection Timeout=30;')
    #     quoted = urllib.parse.quote_plus(conn)
    #     new_con = 'mysql+pymysql:///?odbc_connect={}'.format(quoted)
    #     self.engine = create_engine(new_con, echo=False)
