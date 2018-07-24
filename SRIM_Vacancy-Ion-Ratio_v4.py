# SRIM_Add_N_Plot

import sys, time, re, csv, os, glob, operator, cmath
from datetime import datetime
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def main():    

    comment_csv = "Comment_Table.csv"
    Comment_DF = pd.read_csv(comment_csv)
    
    ColumnInfo = {"Vacancy_File":1, "Range_File": 2, "Offset":3, "Dose":3, "Energy":4, "Tilt":5, "Comment":6, "Include":8, "Set":9}
    
    DF = RatioRowCal(comment_CSV = comment_csv ,ColumnInfo=ColumnInfo)
    
    Set_Value_list = ["None", "Add1", "Add7" ]
#    Set_Value_list = ["Add1", "Add11","Add12"]
    
    frame_list = []
    
    for Set_value in Set_Value_list:
        if Set_value !="None":
            DF_Add = DF[(DF['Set']==Set_value) & (DF['Include']=="Yes")].copy()
            xnew = XminXmax(DF, columnName="Set", columnvalue=Set_value, x_column = "Depth", rowNumber=100)
            DF_sum = AddUP(DF_Add, xnew, SelCol="Set", SelValue=Set_value, ComCol="Comment", Comment_DF= Comment_DF)
        if Set_value == "None":
            DF_sum = DF[(DF['Set']==Set_value) & (DF['Include']=="Yes")].copy()
        
        frame_list.append(DF_sum)
    
    result = pd.concat(frame_list)
    
    result.to_csv("Combined_file.csv", index=False)
    

def AddUP(DF, xnew, SelCol="Set", SelValue="Add", ComCol="Comment", Comment_DF= pd.DataFrame()):
#    DF_Add = DF[DF[SelCol]==SelValue]
    
    DF_Add = DF.groupby([ComCol]).apply( (lambda x: Redepth(x, xnew, xname="Depth", ylist=['Vacancy_by_Ion', 'Vacancy_by_Recoils', 'Ions']) )).reset_index()
    DF_Add.drop(['index'], axis=1, inplace=True)
    DF_sum = DF_Add.groupby([SelCol, 'Depth'])[['Vacancy_by_Ion', 'Vacancy_by_Recoils', 'Ions']].sum().reset_index()
    
    recipename = ''
    ComList = ((Comment_DF[ (Comment_DF[SelCol]==SelValue) & (Comment_DF["Include"]=="Yes") ])[ComCol]).tolist()
    for i in ComList:     
        if i == ComList[-1]:  #this determines whether the fist item is also the last item.
            recipename += i
        else:
            recipename += i + '+'
#    print (recipename)    
    
    DF_sum[ComCol] = recipename
    DF_sum["Vacancy-by-Ion_Range_Ratio"] = DF_sum['Vacancy_by_Ion']/DF_sum['Ions']
    DF_sum["Vacancy-by-Recoils_Range_Ratio"] = DF_sum['Vacancy_by_Recoils']/DF_sum['Ions']

    return DF_sum
    



def XminXmax(dfx, columnName="Set", columnvalue="Add", x_column = "Depth", rowNumber=100):
    DFNew = dfx[dfx[columnName]==columnvalue]
    xmin = DFNew[x_column].min()
    xmax = DFNew[x_column].max()
    xnew = np.linspace(xmin, xmax, num=rowNumber)
    return xnew



def Redepth(dfx, xnew, xname="Depth", ylist=['Vacancy_by_Ion', 'Vacancy_by_Recoils', 'Ions']):
    length = len(dfx.index)
    #print (length)
    #print (len(xnew))
    if (length != len(xnew)):
        print ("length is incorrect")
        return 0
    DF =dfx.copy()
    
    xlist = DF[xname].tolist()
    for item in ylist:
        #print (item)
        ylist = DF[item].tolist()
        ynewCol =[]
        InterpFunc = interp1d(xlist, ylist, kind='linear', bounds_error=False, fill_value= float('nan') )
        ynewCol = InterpFunc(xnew)
        dfx[item] =  np.asarray(ynewCol)
    
    dfx[xname]=np.asarray(xnew)
    
    dfx["Vacancy-by-Ion_Range_Ratio"] = dfx['Vacancy_by_Ion']/dfx['Ions']
    dfx["Vacancy-by-Recoils_Range_Ratio"] = dfx['Vacancy_by_Recoils']/dfx['Ions']
                 
    return dfx     
    



def RatioRowCal(comment_CSV = "Comment_Table.csv" ,ColumnInfo={"Vacancy_File":1, "Range_File": 2, "Offset":3, "Dose":3, "Energy":4, "Tilt":5, "Comment":6, "Include":8}):
    comment = pd.read_csv(comment_CSV )
    #print (comment)
    DF = pd.DataFrame()
    
    for row in comment.iterrows():
       index, data = row
       if data['Include'] == "Yes":
           for k in ColumnInfo.keys():
               ColumnInfo[k] = data[k]
               #print (ColumnInfo[k])
           VacancyDF = FiletoDF(filename=ColumnInfo['Vacancy_File'], datatype="Vacancy", offset=ColumnInfo['Offset'], dose=ColumnInfo['Dose'] )
           RangeDF = FiletoDF(filename=ColumnInfo['Range_File'], datatype="Range", offset=ColumnInfo['Offset'], dose=ColumnInfo['Dose'] )
           RatioDF =  VacancyDF.merge(RangeDF, how="inner", on=['Depth'] )
           RatioDF["Vacancy-by-Ion_Range_Ratio"] = RatioDF['Vacancy_by_Ion']/RatioDF['Ions']
           RatioDF["Vacancy-by-Recoils_Range_Ratio"] = RatioDF['Vacancy_by_Recoils']/RatioDF['Ions']
           for k in ColumnInfo.keys():
               RatioDF[k] = ColumnInfo[k]
           DF = DF.append(RatioDF)
        
    return DF



def RowToDF(ColumnInfo):
    filename = ColumnInfo['Filename']
    datatype = ColumnInfo['datatype']   
    
    LineNumber = LineNumberCal(filename, datatype)
    
    if datatype == "Range" or datatype == "range":
        a = pd.read_table(filename, sep='_', skiprows=LineNumber, engine="python",header=None)
    if datatype == "Vacancy" or datatype == "vacancy":
        a = pd.read_table(filename, sep='_', skiprows=LineNumber, engine="python", skipfooter=1, header=None)
    
    a.columns = ["All"]
    
    replaceSpace = lambda x: pd.Series(re.sub('\s{2,}', ' ', str(x)))
    a["All"] = a["All"].apply(replaceSpace)
    
    foo = lambda x: pd.Series(str(x).split(' '))
    a = a["All"].apply(foo)
    
    if datatype == "Range" or datatype == "range":
        a.rename(columns={0:'Depth',1:'Ions',2:'Recoils'},inplace=True)
    if datatype == "Vacancy" or datatype == "vacancy":
        a.rename(columns={0:'Depth',1:'Vacancy_by_Ion',2:'Vacancy_by_Recoils'},inplace=True)
        a.drop([3], axis=1, inplace=True)
    
    for x in a.columns.tolist():
        a[x] = a[x].astype('float', copy=True)    
    
    a['Depth'] = a['Depth'] - ColumnInfo['Offset']
    
    if datatype == "Range" or datatype == "range":
        a['Ions'] = a['Ions'] * float(ColumnInfo['Dose'])
        a['Recoils'] = a['Recoils'] * float(ColumnInfo['Dose'])
                  
    if datatype == "Vacancy" or datatype == "vacancy":
        a['Vacancy_by_Ion'] = a['Vacancy_by_Ion'] * float(ColumnInfo['Dose']) *1.0E8
        a['Vacancy_by_Recoils'] = a['Vacancy_by_Recoils'] * float(ColumnInfo['Dose'])*1.0E8   
    
    a['Filename'] = filename

    return a

 
def RowToList(DF, ColumnOrder = ['Filename', 'Offset', 'Dose', 'Energy', 'Tilt', 'Comment', 'datatype']):
    DF = DF[ColumnOrder]
    RowList = DF.apply(lambda x: x.tolist(), axis=1)
    return RowList    
    

def FiletoDF(filename, datatype="Range", **kwargs):
    #comment = filename    
    #comment = re.sub(".txt", "", comment) 
    LineNumber = LineNumberCal(filename, datatype)
    
    if datatype == "Range" or datatype == "range":
        a = pd.read_table(filename, sep='_', skiprows=LineNumber, engine="python",header=None)
    if datatype == "Vacancy" or datatype == "vacancy":
        a = pd.read_table(filename, sep='_', skiprows=LineNumber, engine="python", skipfooter=1, header=None)
    
    a.columns = ["All"]
    
    replaceSpace = lambda x: pd.Series(re.sub('\s{2,}', ' ', str(x)))
    a["All"] = a["All"].apply(replaceSpace)
    
    foo = lambda x: pd.Series(str(x).split(' '))
    a = a["All"].apply(foo)
    
    if datatype == "Range" or datatype == "range":
        a.rename(columns={0:'Depth',1:'Ions',2:'Recoils'},inplace=True)
    if datatype == "Vacancy" or datatype == "vacancy":
        a.rename(columns={0:'Depth',1:'Vacancy_by_Ion',2:'Vacancy_by_Recoils'},inplace=True)
        a.drop([3], axis=1, inplace=True)
    
    for x in a.columns.tolist():
        a[x] = a[x].astype('float', copy=True)    
       
    
    for key, value in kwargs.items():
        #print (key)
        if (key == 'offset')and(len(kwargs.items())!=0):       
            a['Depth'] = a['Depth'] - kwargs[key]
            
        if (key == 'dose')and(len(kwargs.items())!=0):
            #a['comment'] = comment + '_' + NumberToSciStr(kwargs[key])
            if datatype == "Range" or datatype == "range":
                a['Ions'] = a['Ions'] * kwargs[key]
                a['Recoils'] = a['Recoils'] * kwargs[key]
                  
            if datatype == "Vacancy" or datatype == "vacancy":
                a['Vacancy_by_Ion'] = a['Vacancy_by_Ion'] * kwargs[key] *1.0E8
                a['Vacancy_by_Recoils'] = a['Vacancy_by_Recoils'] * kwargs[key]*1.0E8    

    return a


def NumberToSciStr(number):
    return str('{:.2e}'.format(number))


def Range_FiletoDF(filename, datatype="Range"):
    comment = filename    
    re.sub(".txt", "", comment) 

    LineNumber = LineNumberCal(filename, datatype)

    a = pd.read_table(filename, sep='_', skiprows=LineNumber, engine="python",header=None)
    a.columns = ["All"]

    replaceSpace = lambda x: pd.Series(re.sub('\s{2,}', ' ', str(x)))
    a["All"] = a["All"].apply(replaceSpace)

    foo = lambda x: pd.Series(str(x).split(' '))
    a = a["All"].apply(foo)
       
    a.rename(columns={0:'Depth',1:'Ions',2:'Recoils'},inplace=True)
    a['comment'] = comment
    return a

    
        
def Vacancy_FiletoDF(filename, datatype="vacancy"):
    comment = filename    
    re.sub(".txt", "", comment) 
    #print (filename)    
    LineNumber = LineNumberCal(filename, datatype)
    a = pd.read_table(filename, sep='_', skiprows=LineNumber, engine="python", skipfooter=1, header=None)
    a.columns = ["All"]
    replaceSpace = lambda x: pd.Series(re.sub('\s{2,}', ' ', str(x)))
    a["All"] = a["All"].apply(replaceSpace)
    #split column
    foo = lambda x: pd.Series(str(x).split(' '))
    a = a["All"].apply(foo)
    
    a.rename(columns={0:'Depth',1:'Vacancy_by_Ion',2:'Vacancy_by_Recoils'},inplace=True)
    a.drop([3], axis=1, inplace=True)
    a['comment'] = comment
    return a


def LineNumberCal(filename, datatype = "Vacancy" ):
    if datatype == "Vacancy" or datatype == "vacancy":
        pattern = '-----------  -----------  ------------'
    if datatype == "Range" or datatype == "range":
        pattern = '-----------  ----------  ------------'
    #print (filename)
    with open(filename) as myFile:
        for num, line in enumerate(myFile, 1):
            #print (line)
            if pattern in line:
                #print (line)
                #print (num)
                return num

def CombineMgage(waferinfo, points=9):
    newDF = pd.DataFrame()
    cwd = os.getcwd()
    os.chdir(cwd)
    for file in glob.glob("*.txt"):
        data= MgageExtract(file, waferinfo, points=9)
        newDF=newDF.append(data)
    return newDF

def MgageExtract(txtfile, waferinfo, points=9):
    
    df = pd.read_table(txtfile, sep=',|\t',  engine='python', skiprows=1)    
    df = df.rename(columns={'Ohm/sq': 'Rsheet', '  Angstr': 'Thickness'})    
    data = df[0:points]    
    data=data[["Rsheet"]]    
    data["site"]=range(1,1+points)
    data["filename"]=txtfile  
    data = data.merge(waferinfo, on=["filename"], how="inner")
    return data
    


def fT_Cal_df(df, groupby_list):
    fT_df = df.groupby(groupby_list).apply( (lambda x: fT_Linear_fit(x) )).reset_index()
    return fT_df

def fT_Linear_fit(df):
    y=np.log10(df.H21_Mag)
    x=np.log10(df.frequency)
    X = sm.add_constant(x)
    model= sm.OLS( y, X )
    result=model.fit()
    constval = result.params[0]
    slope = result.params[1]
    rsquare= result.rsquared
    fT_log= -constval/slope
    fT= pow(10.0, fT_log)
    return pd.Series({'fT' : fT,  'R^2': rsquare, 'slope':slope, 'fT_log':fT_log, 'intercept':constval})    
    
    
    
def GetParameterList(FileAddress):
    lines = [line.rstrip('\n') for line in open(FileAddress)]
    return lines


def MAGCal(row):
    D = row['S11'] * row['S22'] - row['S12'] * row['S21']    
    K= (1- abs(pow(row['S11'], 2)) - abs(pow(row['S22'], 2)) + abs(pow(D,2))) / (2*abs(row['S12']*row['S21']))
    MSG = abs (row['S21'] / row['S12']) 
    #This function should be applied only if K>1
    if K >=1:
        MAG = abs ( MSG * (K- cmath.sqrt(K*K -1)) )
        return MAG
    else:
        return None


def MSGCal(row):
    MSG = abs (row['S21'] / row['S12'])    
    return MSG
    
def KCal(row):
    D = row['S11'] * row['S22'] - row['S12'] * row['S21']    
    K= (1- abs(pow(row['S11'], 2)) - abs(pow(row['S22'], 2)) + abs(pow(D,2))) / (2*abs(row['S12']*row['S21']))
    return K

def Y11Cal(row):
    Y0= 1/50.0
    D=(1+row['S11'])*(1+row['S22'])-row['S12']*row['S21']
    Y11= Y0*((1-row['S11'])*(1+row['S22'])+row['S12']*row['S21'])/D
    return Y11

def Y12Cal(row):
    Y0= 1/50.0
    D=((1+row['S11'])*(1+row['S22'])-row['S12']*row['S21'])
    Y12= -2 * Y0 * row['S12'] / D
    return Y12

def Y21Cal(row):
    Y0= 1/50.0
    D=((1+row['S11'])*(1+row['S22'])-row['S12']*row['S21'])
    Y21= -2 * Y0 * row['S21'] / D
    return Y21

def Y22Cal(row):
    Y0= 1/50.0
    D=((1+row['S11'])*(1+row['S22'])-row['S12']*row['S21'])
    Y22= Y0*((1+row['S11'])*(1-row['S22'])+row['S12']*row['S21'])/D
    return Y22

def CJSCal(row):
    pi=3.1415926
    comp = 1.0 / (row['Y22']+row['Y12'])
    denominator = comp.imag*2*pi*row['Frequency']  #frequency in Ghz, not in Hz
    CJS_fF = -1.0E6 / denominator
    return CJS_fF


def ReadSParameter(S_File):
    a = pd.read_table(S_File, delim_whitespace=True)
   
    
    return a
    
    

def ProcessDataFrame(a):
    a['S11'] = a.apply(NewComplexS11, axis=1)
    a['S12'] = a.apply(NewComplexS12, axis=1)
    a['S21'] = a.apply(NewComplexS21, axis=1)
    a['S22'] = a.apply(NewComplexS22, axis=1)
    
    a = a.drop('S11R', 1)
    a = a.drop('S11I', 1)
    a = a.drop('S12R', 1)
    a = a.drop('S12I', 1)
    a = a.drop('S21R', 1)
    a = a.drop('S21I', 1)
    a = a.drop('S22R', 1)
    a = a.drop('S22I', 1)
    
    a['K'] = a.apply(KCal, axis=1)
    a['MSG'] = a.apply(MSGCal, axis=1)    
    a['MAG'] = a.apply(MAGCal, axis=1)
    
    a['Y11'] = a.apply(Y11Cal, axis=1)
    a['Y12'] = a.apply(Y12Cal, axis=1)
    a['Y21'] = a.apply(Y21Cal, axis=1)
    a['Y22'] = a.apply(Y22Cal, axis=1)
       
    return a



def NewComplexS22(row):
    return np.complex (row['S22R'] , row['S22I'])


def NewComplexS21(row):
    return np.complex (row['S21R'] , row['S21I'])


def NewComplexS12(row):
    return np.complex (row['S12R'] , row['S12I'])


def NewComplexS11(row):
    return np.complex (row['S11R'] , row['S11I'])
 

def get_num(x):
    return float(''.join(ele for ele in x if ele.isdigit() or ele == '.'))
    

def AddFabInfo(M1Data_Orig, M1Data_Final, FabInfo):
    M1Orig = pd.read_csv(M1Data_Orig)
    M1Orig['Wafer_ID']= M1Orig['Wafer_ID'].str.strip()
    
    
    a = pd.read_csv(FabInfo, names= ['FabEvent', 'LotID', 'Wafer_ID', 'D', 'E', 'Description','G', 'H', 'I', 'J'])
    a['FabEvent'] = a['FabEvent'].str.strip()
    a['LotID']= a['LotID'].str.strip()
    a['Wafer_ID']= a['Wafer_ID'].str.strip()
    fabdata=a[['FabEvent', 'Wafer_ID', 'Description']]    
    
    aaa = pd.merge(M1Orig, fabdata, on=['Wafer_ID'], how='outer')    
    #aaa.to_csv(M1Data_Final, index=False)
    
    return aaa



if __name__ == '__main__':
    main()
