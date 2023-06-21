from src.data_io.data_io import DataIO
from src.sql_connection import SQLConnection

sql_connection = SQLConnection("db4free")
data_io = DataIO.create(
    sql_connection=sql_connection,
    load_from_sql=True,
    save_to_sql=False,
)

print(data_io.loader.load_something("select * from channel"))
