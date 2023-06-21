from src.data_io.data_io import DataIO
from src.sql_connection import SQLConnection

sql_connection = SQLConnection("db4free")
data_io = DataIO.create(
    sql_connection=sql_connection,
    load_from_sql=True,
    save_to_sql=False,
)

# print(data_io.loader.load_something("select * from channel"))

print(data_io.loader.load_panel_data(1, [1, 2, 3, 4], "2023-06-20", "2023-07-01").sort_values("ts", ascending=False).head())

# extra_clauses: list[str] = ["1=1"]
#
# extra_clauses = ["and " + clause for clause in extra_clauses]
# extra_clauses = " ".join(extra_clauses)
#
# print(extra_clauses)
