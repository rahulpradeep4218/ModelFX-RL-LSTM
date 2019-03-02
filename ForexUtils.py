import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import select
from sqlalchemy import or_
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler


##### Define table where ML Variables are stored
varTableName = "mlvariables"


def getSQLEngine():
    ForexDBParameters = dict(line.strip().split('=') for line in open('forexDB.properties') if not line.startswith('#') and not line.startswith('\n'))
    engine = create_engine('mysql+pymysql://' + ForexDBParameters['username'] + ':' + ForexDBParameters['password'] + '@localhost:3306/forex')
    return engine

commonSQLEngine = getSQLEngine()

def getMLVariables():
    dict = {}
    sqlEngine = getSQLEngine()
    varDF = pd.read_sql_query("SELECT * FROM " + varTableName + " WHERE Active = '1'", sqlEngine)
    for index,row in varDF.iterrows():
        dict[row['Name']] = str(row['Value'])

    return dict


def getCategoricalColumnsFromDF(inputDF, colName, numClasses, append=True, dropCol=True):
    catColsDF = pd.DataFrame(to_categorical(inputDF[colName], num_classes=numClasses), index=inputDF.index)
    colArr = []
    catColsDF = catColsDF.astype('float64')
    for i in range(numClasses):
        colArr.append(colName + str(i))
    catColsDF.columns = [colArr]
    # for col in catColsDF.columns:
    if append:
        inputDF[colArr] = catColsDF[colArr]
        catColsDF = inputDF
    if dropCol:
        catColsDF.drop([colName], axis=1, inplace=True)
    return (catColsDF,colArr)

#### Function to get count of number of records in given table
def count_records(table_name,condition):
    try:
        sqlEng = getSQLEngine()
        resultDF = pd.read_sql_query("SELECT COUNT(*) FROM " + table_name + " WHERE " + condition,sqlEng)
        count = int(resultDF.iloc[0,0])
    except:
        count = 0
    return count


#### Function to drop a table in MySQL DB
def drop_table(table_name):
    sqlEng = getSQLEngine()
    sqlEng.execute("DROP TABLE IF EXISTS " + table_name)


#### Function to create a table in MySQL
"""
actions_table_details = {'name':'metaactions', 'col':['Action', 'Time'], 'type':['VARCHAR(20)', 'DATETIME'], 'null': [False,False]}
"""
def create_table_db(table_details):
    action_table_sql = "CREATE TABLE IF NOT EXISTS `forex`.`" + table_details['name'] + "` ( `Id` INT NOT NULL AUTO_INCREMENT "
    for count, col in enumerate(table_details['col']):
        action_table_sql = action_table_sql + " , `" + col + "` " + table_details['type'][count]
        if table_details['null'][count]:
            action_table_sql = action_table_sql + " NULL"
        else:
            action_table_sql = action_table_sql + " NOT NULL"
    action_table_sql += " , PRIMARY KEY (`Id`)) ENGINE=INNODB"
    print("Create Table SQL : ",action_table_sql)
    execute_query_db(action_table_sql)

def execute_query_db(sql_query, sqlEng=0):
    if sqlEng == 0:
        sqlEng = getSQLEngine()
    sqlEng.execute(sql_query)

def getTableRows_db(query):
    rs = commonSQLEngine.execute(query)
    return rs

def getNormalizedData(dataReference):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(dataReference)
    return scaler


def getOneHotEncoding(x, num_classes):
    return np.eye(num_classes)[x]

def deleteFromList(x, indexList):
    for ind in sorted(indexList, reverse=True):
        del x[ind]

    return x