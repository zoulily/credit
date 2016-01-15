import numpy as np
import pandas as pd
basic = pd.read_csv('./Data/basic_information.csv')
basic = basic.iloc[:,3:]
borinfo = pd.read_csv('./Data/borrower_information.csv')
borinfo = borinfo.iloc[:,3:]
borlenhis = pd.read_csv('./Data/borrowing_lending_history.csv')
borlenhis = borlenhis.iloc[:,3:]
eninfo = pd.read_csv('./Data/enterprise_information.csv')
eninfo = eninfo.iloc[:,3:]
guainfo = pd.read_csv('./Data/guarantee_information.csv')
guainfo = guainfo.iloc[:,3:]
homeinfo = pd.read_csv('./Data/home_mortgage_information.csv')
homeinfo = homeinfo.iloc[:,3:]
referprice = pd.read_csv('./Data/reference_pricing.csv')
referprice = referprice.iloc[:,3:]

result = pd.concat([basic,borinfo,borlenhis,eninfo,guainfo,homeinfo,referprice],axis=1)

result.to_csv('./Data/housing_source.csv',index=False)