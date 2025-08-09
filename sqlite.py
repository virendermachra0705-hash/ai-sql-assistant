import sqlite3
## coonect to sqlite

connection=sqlite3.connect("student.db")


## create a cursor object to insert record,create table
cursor=connection.cursor()

table_info="""
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25),MARKS INT)


"""

cursor.execute(table_info)


## insert some records
cursor.execute(''' Insert into Student values('krish','data science','A',90)''')
cursor.execute(''' Insert into Student values('Virender','data science','A',100)''')
cursor.execute(''' Insert into Student values('Rahul','Cloud','A',97)''')
cursor.execute(''' Insert into Student values('Mohit','Income Tax','A',86)''')
cursor.execute(''' Insert into Student values('Rohit','Cloud','A',91)''')


## display all the records
print("the inserted records are")
data=cursor.execute("select * from Student")
for row in data:
    print(row)

## commit your changes in database
connection.commit()
connection.close()
