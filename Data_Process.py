import pandas as pd
import numpy as np
import pickle

import ForexUtils as FXUtils

class ProcessedData:
    def __init__(self, inputSymbol, train_test_predict ="train"):
        sqlEngine = FXUtils.getSQLEngine()
        mlVariables = FXUtils.getMLVariables()
        limitTable = int(mlVariables['LimitMLTrainTable'])
        self.symbol = inputSymbol
        self.train_test_split_ratio = float(mlVariables['RL-TrainTestSplit'])
        self.train_test_predict = train_test_predict

        if train_test_predict == "train" or train_test_predict == "test":
            if limitTable == 0:
                tableSize = FXUtils.count_records(mlVariables['MLDataTableName'], "Symbol = '" + inputSymbol + "'")
            else:
                tableSize = limitTable
            splitPoint = int(tableSize * self.train_test_split_ratio)
            recordsAfterSplit = tableSize - splitPoint
            print("table size : {0}, split point : {1}, records after split : {2}".format(tableSize, splitPoint, recordsAfterSplit))

        if train_test_predict == "train":
            tableName = mlVariables['MLDataTableName']
            sqlQuery = "SELECT * FROM " + tableName + " WHERE Symbol = '" + inputSymbol + "' ORDER BY Time LIMIT " + str(splitPoint)
            symbolPickle = {}
            symbolPickle['scalers'] = {}
            self.pickle = symbolPickle

        elif train_test_predict == "test":
            tableName = mlVariables['MLDataTableName']
            sqlQuery = "SELECT * FROM " + tableName + " WHERE Symbol = '" + inputSymbol + "' ORDER BY Time LIMIT " + str(splitPoint) + "," + str(recordsAfterSplit)


        else:
            tableName = mlVariables['MLPredTableName']
            sqlQuery = "SELECT * FROM " + tableName + " WHERE Symbol = '"+inputSymbol+"' AND Status = 'DATA_READY'"


        rawData = pd.read_sql_query(sqlQuery,sqlEngine)
        rawData = rawData.set_index('Time')
        rawData['CloseShifted'] = rawData['Close'].shift(1)
        thresh = 0.8 * len(rawData)
        rawData = rawData.dropna(axis=1, thresh=thresh)
        rawData = rawData.dropna(axis=0)
        print("raw data 1st : ", rawData.head(1))

        df = rawData[['CloseShifted']].copy()
        df['Bar_HC'] = rawData['High'] - rawData['Close']
        df['Bar_HO'] = rawData['High'] - rawData['Open']
        df['Bar_HL'] = rawData['High'] - rawData['Low']
        df['Bar_CL'] = rawData['Close'] - rawData['Low']
        df['Bar_CO'] = rawData['Close'] - rawData['Open']
        df['Bar_OL'] = rawData['Open'] - rawData['Low']
        df['Volume'] = rawData['Volume']

        self.df = df
        self.rawData = rawData
        self.split_point = splitPoint

    def addSimpleFeatures(self):
        mlVariables = FXUtils.getMLVariables()
        addOnFeaturesList = mlVariables['Model5AddonFeatures'].split(',')
        addOnNumericalExcepList = mlVariables['Model5AddonFeaturesNumerical'].split(',')
        addOnNumClassesList = mlVariables['Model5AddonFeaturesNumClasses'].split(',')
        print("features : ", addOnFeaturesList)
        print("numClasses : ", addOnNumClassesList)
        for i, col in enumerate(addOnFeaturesList):
            if col not in addOnNumericalExcepList:
                self.df[col] = self.rawData[col]
                self.df, colArr = FXUtils.getCategoricalColumnsFromDF(self.df, col, int(addOnNumClassesList[i]))
            else:
                self.df[col] = self.rawData[col]
                self.df[col] = self.df[col].astype('float64')

    def apply_normalization(self):
        if self.train_test_predict == "train":
            for col in self.df.columns.values.tolist():
                self.pickle['scalers'][col] = FXUtils.getNormalizedData(self.df[[col]])
                self.df[col] = self.pickle['scalers'][col].transform(self.df[[col]])
            pickle.dump(self.pickle, open("Pickles\\" + self.symbol + "-Pickle.pkl", "wb"))
        elif self.train_test_predict == "test":
            symbolPickle = pickle.load(open("Pickles\\" + self.symbol + "-Pickle.pkl", "rb"))
            for col in self.df.columns.values.tolist():
                self.df[col] = symbolPickle['scalers'][col].transform(self.df[[col]])



# testSymbol = "EURUSD"
# forexData = ProcessedData(testSymbol)
# forexData.addSimpleFeatures()
# print("Dataframe : ",forexData.df)

##############