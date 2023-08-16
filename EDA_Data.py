import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

## list of csv for healthcare provider fraud detection.
list_of_csv=os.listdir()
# print(list_of_csv)

"""### Loading Train and Test data."""

# train data
# train_label=pd.read_csv('https://drive.google.com/uc?id=1RpIT7D5Omw2Wk2xyPT15Qm-M4s1j7V_J')
# train_beneficiary=pd.read_csv('https://drive.google.com/uc?id=1cskBayyKjjFI-RqUACejlw54P0Xnp0xZ')
# train_inpatient=pd.read_csv('https://drive.google.com/uc?id=1s1C4N3PJHobQ3Rz7fjcD6Am9F3iBg4ap')
# train_outpatient=pd.read_csv('https://drive.google.com/uc?id=1m0oVVMZvaj2geEp2v8GqGirs9Y1P5BCy')

# # test data
# test_label=pd.read_csv('https://drive.google.com/uc?id=1Q07zTwCIBdTWWg2S-F1fovm8OTttqT79')
# test_beneficiary=pd.read_csv('https://drive.google.com/uc?id=1A3VXWP5KyaTYeDFbB0y47Zt6wVQ057j8')
# test_inpatient=pd.read_csv('https://drive.google.com/uc?id=1ORJNu4_XTDa1oX-QciJauuaIfB6tMPVh')
# test_outpatient=pd.read_csv('https://drive.google.com/uc?id=1ORJNu4_XTDa1oX-QciJauuaIfB6tMPVh')


# # train data
# train_label=pd.read_csv('Dataset/Train-1542865627584.csv')
# train_beneficiary=pd.read_csv('Dataset/Train_Beneficiarydata-1542865627584.csv')
# train_inpatient=pd.read_csv('Dataset/Train_Inpatientdata-1542865627584.csv')
# train_outpatient=pd.read_csv('Dataset/Train_Outpatientdata-1542865627584.csv')

# # test data
# test_label=pd.read_csv('Dataset/Test-1542969243754.csv')
# test_beneficiary=pd.read_csv('Dataset/Test_Beneficiarydata-1542969243754.csv')
# test_inpatient=pd.read_csv('Dataset/Test_Inpatientdata-1542969243754.csv')
# test_outpatient=pd.read_csv('Dataset/Test_Outpatientdata-1542969243754.csv')

# train data
train_label=pd.read_csv('data/train-label.csv')
train_beneficiary=pd.read_csv('data/train-bene.csv')
train_inpatient=pd.read_csv('data/train-inp.csv')
train_outpatient=pd.read_csv('data/train-out.csv')

# test data
test_label=pd.read_csv('data/test-label.csv')
test_beneficiary=pd.read_csv('data/test-bene.csv')
test_inpatient=pd.read_csv('data/test-inp.csv')
test_outpatient=pd.read_csv('data/test-out.csv')

"""## EDA on train data

### Trian label
"""

plt.figure(figsize=(7,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.countplot(data=train_label,x='PotentialFraud') # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of  Class Labels', fontsize=20)
plt.xlabel('Whether Potentially Fraud', size = 14)
plt.ylabel('Count of fraud', size = 14)
count=train_label['PotentialFraud'].value_counts()
no=np.round((count[0]/(count[0]+count[1]))*100,4)
yes=np.round((count[1]/(count[0]+count[1]))*100,4)
print('percentage of Fradus in data ', yes)
print('percentage of Non Frauds in data ', no)
plt.show()

"""### Observations
 1.There is no missing valus in class labels(Potential_frauds)
 
 2.Data is highly imbanced
"""


train_label['PotentialFraud']=train_label['PotentialFraud'].replace('Yes',1)
train_label['PotentialFraud']=train_label['PotentialFraud'].replace('No',0)
print(train_label.head())



"""## Beneficiary data"""

train_bene_col=train_beneficiary.columns
for i in train_bene_col:
    print(i,"=",train_beneficiary[i].isnull().any())

train_bene_col=train_beneficiary.columns
train_bene_col=list(train_bene_col)
train_bene_col=train_bene_col[3:-4] # removing DOB,DOD and other continues feateus.
for i in train_bene_col:
    print(i,"=",train_beneficiary[i].unique())

"""

### Preprocessing


1. only data of death column has Nan values 
2. we have to change gender to 1 and 0 from 1 and 2.(for  easy interpritaion)
3. In RenalDiseaseIndicator we have to replace Y with 1.
4. we have to make weather a beneficiary has chronic condition or not by replacing 2 with 0 (no cronic condition).From this we can add another feature called Total_chronic_condition by simply adding them."""

# replacing 2 with 0 in gender, train_beneficary[gender]
train_beneficiary['Gender']=train_beneficiary['Gender'].replace(2,0)
train_beneficiary['Gender'].head()
# replacing in test beneficary data
test_beneficiary['Gender']=test_beneficiary['Gender'].replace(2,0)
test_beneficiary['Gender'].head()

# ditribution plot of gender
plt.figure(figsize=(7,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.countplot(data=train_beneficiary,x='Gender') # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of Gender', fontsize=20)
plt.xlabel('Gender', size = 14)
plt.ylabel('Count', size = 14)
count=train_beneficiary['Gender'].value_counts()
no=np.round((count[0]/(count[0]+count[1]))*100,4)
yes=np.round((count[1]/(count[0]+count[1]))*100,4)
print('percentage of gender 1 ', yes)
print('percentage of gender O ', no)
plt.show()


# replacing Y with 1 in RenalDiseaseIndicator
train_beneficiary['RenalDiseaseIndicator']=train_beneficiary['RenalDiseaseIndicator'].replace('Y',1)
# repacling in test data
test_beneficiary['RenalDiseaseIndicator']=test_beneficiary['RenalDiseaseIndicator'].replace('Y',1)
test_beneficiary['RenalDiseaseIndicator'].head()


# replacing 2, with 0 in all 10 chronic condition, here we considering zero as no chornic condition
train_beneficiary['ChronicCond_Alzheimer']=train_beneficiary['ChronicCond_Alzheimer'].replace(2,0)
train_beneficiary['ChronicCond_Cancer']=train_beneficiary['ChronicCond_Cancer'].replace(2,0)
train_beneficiary['ChronicCond_Depression']=train_beneficiary['ChronicCond_Depression'].replace(2,0)
train_beneficiary['ChronicCond_Diabetes']=train_beneficiary['ChronicCond_Diabetes'].replace(2,0)
train_beneficiary['ChronicCond_Heartfailure']=train_beneficiary['ChronicCond_Heartfailure'].replace(2,0)
train_beneficiary['ChronicCond_IschemicHeart']=train_beneficiary['ChronicCond_IschemicHeart'].replace(2,0)
train_beneficiary['ChronicCond_KidneyDisease']=train_beneficiary['ChronicCond_KidneyDisease'].replace(2,0)
train_beneficiary['ChronicCond_ObstrPulmonary']=train_beneficiary['ChronicCond_ObstrPulmonary'].replace(2,0)
train_beneficiary['ChronicCond_Osteoporasis']=train_beneficiary['ChronicCond_Osteoporasis'].replace(2,0)
train_beneficiary['ChronicCond_rheumatoidarthritis']=train_beneficiary['ChronicCond_rheumatoidarthritis'].replace(2,0)
train_beneficiary['ChronicCond_stroke']=train_beneficiary['ChronicCond_stroke'].replace(2,0)

# replacing in test data
test_beneficiary['ChronicCond_Alzheimer']=test_beneficiary['ChronicCond_Alzheimer'].replace(2,0)
test_beneficiary['ChronicCond_Cancer']=test_beneficiary['ChronicCond_Cancer'].replace(2,0)
test_beneficiary['ChronicCond_Depression']=test_beneficiary['ChronicCond_Depression'].replace(2,0)
test_beneficiary['ChronicCond_Diabetes']=test_beneficiary['ChronicCond_Diabetes'].replace(2,0)
test_beneficiary['ChronicCond_Heartfailure']=test_beneficiary['ChronicCond_Heartfailure'].replace(2,0)
test_beneficiary['ChronicCond_IschemicHeart']=test_beneficiary['ChronicCond_IschemicHeart'].replace(2,0)
test_beneficiary['ChronicCond_KidneyDisease']=test_beneficiary['ChronicCond_KidneyDisease'].replace(2,0)
test_beneficiary['ChronicCond_ObstrPulmonary']=test_beneficiary['ChronicCond_ObstrPulmonary'].replace(2,0)
test_beneficiary['ChronicCond_Osteoporasis']=test_beneficiary['ChronicCond_Osteoporasis'].replace(2,0)
test_beneficiary['ChronicCond_rheumatoidarthritis']=test_beneficiary['ChronicCond_rheumatoidarthritis'].replace(2,0)
test_beneficiary['ChronicCond_stroke']=test_beneficiary['ChronicCond_stroke'].replace(2,0)


train_beneficiary['ChronicCond_Alzheimer'].unique()


plt.figure(figsize=(7,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.countplot(data=train_beneficiary,x='Race') # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of Race', fontsize=20)
plt.xlabel('Race', size = 14)
plt.ylabel('Count', size = 14)
count=train_beneficiary['Gender'].value_counts()
plt.show()


# county distribution
plt.figure(figsize=(16,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.countplot(data=train_beneficiary,x='State') # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of Country', fontsize=20)
plt.xlabel('State', size = 14)
plt.ylabel('Count', size = 14)
count=train_beneficiary['Gender'].value_counts()
plt.show()


plt.figure(figsize=(16,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.distplot(train_beneficiary['IPAnnualReimbursementAmt']) # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of IPAnnualReimbursementAmt', fontsize=20)
plt.xlabel('IPAnnualReimbursementAmt', size = 14)
plt.ylabel('Density', size = 14)
plt.show()



for i in range(0,101,10):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["IPAnnualReimbursementAmt"],i),))

for i in range(90,101,1):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["IPAnnualReimbursementAmt"],i),))

plt.figure(figsize=(7,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.boxplot(y=train_beneficiary['IPAnnualReimbursementAmt']) # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of IPAnnualReimbursementAmt', fontsize=20)
plt.xlabel('IPAnnualReimbursementAmt', size = 14)
plt.ylabel('Percentile', size = 14)
plt.show()

# IQR
for i in range(0,100,25):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["IPAnnualReimbursementAmt"],i)))

"""### Observations
1. diffrence between 99th percentile and 100th percentile is very big.
2. 80% of benefirary got reimbursent less then equal to 5000

##### checking of outliers  in annualReimbursement
"""

for i in np.arange(99,100.1,0.1):
    print('persentle {0} is {1}'.format(i,np.percentile(np.absolute(train_beneficiary["IPAnnualReimbursementAmt"]),i)))

"""1. AS we can see 100 percentile is double of 99.9 percentlile. from this we can conclued that it may be an outlier."""




plt.figure(figsize=(16,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.distplot(train_beneficiary['IPAnnualDeductibleAmt'])  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of IPAnnualDeductibleAmt', fontsize=20)
plt.xlabel('IPAnnualDeductibleAmt', size = 14)
plt.ylabel('Density', size = 14)
plt.show()



for i in range(0,101,10):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["IPAnnualDeductibleAmt"],i),))

for i in range(90,101,1):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["IPAnnualDeductibleAmt"],i),))

plt.figure(figsize=(7,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.boxplot(y=train_beneficiary['IPAnnualDeductibleAmt']) # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of IPAnnualDeductibleAmt', fontsize=20)
plt.xlabel('IPAnnualDeductibleAmt', size = 14)
plt.ylabel('Percentile', size = 14)
plt.show()

# IQR
for i in range(0,100,25):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["IPAnnualDeductibleAmt"],i),))

"""##### checking for outliers"""

for i in np.arange(99,100.1,0.1):
    print('persentle {0} is {1}'.format(i,np.percentile(np.absolute(train_beneficiary["IPAnnualDeductibleAmt"]),i)))
    

plt.figure(figsize=(16,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.distplot(train_beneficiary['OPAnnualReimbursementAmt']) # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of 0PAnnualReimbursementAmt', fontsize=20)
plt.xlabel('OPAnnualReimbursementAmt', size = 14)
plt.ylabel('Density', size = 14)
plt.show()



for i in range(0,101,10):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["OPAnnualReimbursementAmt"],i),))

for i in range(90,101,1):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["OPAnnualReimbursementAmt"],i),))

plt.figure(figsize=(7,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.boxplot(y=train_beneficiary['OPAnnualReimbursementAmt']) # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of OPAnnualReimbursementAmt', fontsize=20)
plt.xlabel('OPAnnualReimbursementAmt', size = 14)
plt.ylabel('Percentile', size = 14)
plt.show()

for i in range(0,101,25):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["OPAnnualReimbursementAmt"],i),))

# chekcing for outliers
for i in np.arange(99,100.1,0.1):
    print('persentle {0} is {1}'.format(i,np.percentile(np.absolute(train_beneficiary["OPAnnualReimbursementAmt"]),i)))



plt.figure(figsize=(16,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.distplot(train_beneficiary['OPAnnualDeductibleAmt'])  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of OPAnnualDeductibleAmt', fontsize=20)
plt.xlabel('OPAnnualDeductibleAmt', size = 14)
plt.ylabel('Density', size = 14)
plt.show()


for i in range(0,101,10):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["OPAnnualDeductibleAmt"],i),))

for i in range(90,101,1):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["OPAnnualDeductibleAmt"],i),))

plt.figure(figsize=(7,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.boxplot(y=train_beneficiary['OPAnnualDeductibleAmt']) # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of OPAnnualDeductibleAmt', fontsize=20)
plt.xlabel('OPAnnualDeductibleAmt', size = 14)
plt.ylabel('Percentile', size = 14)
plt.show()

for i in range(0,101,25):
    print('persentle {0} is {1}'.format(i,np.percentile(train_beneficiary["OPAnnualDeductibleAmt"],i),))

# chekcing for outliers
for i in np.arange(99,100.1,0.1):
    print('persentle {0} is {1}'.format(i,np.percentile(np.absolute(train_beneficiary["OPAnnualDeductibleAmt"]),i)))


plt.figure(figsize=(30,20))
corrMatrix = train_beneficiary.corr()
hm = sns.heatmap(corrMatrix, annot = True)
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12)
plt.show()




plt.figure(figsize=(30,20))
corrMatrix = train_beneficiary.corr(method='spearman')
hm = sns.heatmap(corrMatrix, annot = True)
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12)
plt.show()



plt.figure(figsize=(12,6))
sns.scatterplot(x='IPAnnualReimbursementAmt',y='IPAnnualDeductibleAmt',data=train_beneficiary)
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('IPAnnualReimbursementAmt vs IPAnnualDeductibleAmt', fontsize=20)
plt.xlabel('IPAnnualReimbursementAmt', size = 14)
plt.ylabel('IPAnnualDeductibleAmt', size = 14)
plt.show()


train_outpatient.head()

train_outpatient.info()


train_outpatient['InscClaimAmtReimbursed'].isnull().any()

plt.figure(figsize=(16,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.distplot(train_outpatient['InscClaimAmtReimbursed'])  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of InscClaimAmtReimbursed', fontsize=20)
plt.xlabel('InscClaimAmtReimbursed', size = 14)
plt.ylabel('Density', size = 14)
plt.show()

for i in range(0,101,25):
    print('persentle {0} is {1}'.format(i,np.percentile(train_outpatient["InscClaimAmtReimbursed"],i),))

for i in range(0,101,10):
    print('persentle {0} is {1}'.format(i,np.percentile(train_outpatient["InscClaimAmtReimbursed"],i),))

for i in range(90,101,1):
    print('persentle {0} is {1}'.format(i,np.percentile(train_outpatient["InscClaimAmtReimbursed"],i),))

for i in np.arange(99,100.1,0.1):
    print('persentle {0} is {1}'.format(i,np.percentile(np.absolute(train_outpatient["InscClaimAmtReimbursed"]),i)))

plt.figure(figsize=(7,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.boxplot(y=train_outpatient['InscClaimAmtReimbursed']) # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of InscClaimAmtReimbursed', fontsize=20)
plt.xlabel('InscClaimAmtReimbursed', size = 14)
plt.ylabel('Percentile', size = 14)
plt.show()



# There are some data poins which are na becasue of that they are not attended by physicain or some hunam error.

train_outpatient['AttendingPhysician'].isnull().any()
train_outpatient['AttendingPhysician']=train_outpatient['AttendingPhysician'].fillna(0)
train_outpatient['AttendingPhysician'].isnull().any()
# filling nan values in test data
test_outpatient['AttendingPhysician']=test_outpatient['AttendingPhysician'].fillna(0)

# county distribution
plt.figure(figsize=(20,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.countplot(data=train_outpatient,x='AttendingPhysician',order=train_outpatient['AttendingPhysician'].value_counts()[:20].index) # name of the category(index)  
plt.xticks(size = 10) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of AttendingPhysician', fontsize=20)
plt.xlabel('AttendingPhysician', size = 14)
plt.ylabel('Count', size = 14)
plt.show()

train_outpatient['AttendingPhysician'].value_counts()[:20].index



# checking nan values 
train_outpatient['OperatingPhysician'].isnull().any()
train_outpatient['OperatingPhysician']=train_outpatient['OperatingPhysician'].fillna(0)
train_outpatient['OperatingPhysician'].isnull().any()
# removing na from test data
test_outpatient['OperatingPhysician']=test_outpatient['OperatingPhysician'].fillna(0)

#counting the physican code 
operating_phy=train_outpatient['OperatingPhysician'].value_counts()[1:21]

# county distribution
plt.figure(figsize=(20,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
# sns.barplot(operating_phy.index,operating_phy.values,order=operating_phy.index) # name of the category(index)  

df = pd.DataFrame({'OperatingPhysician': operating_phy.index, 'Count': operating_phy.values})
sns.barplot(data=df, x='OperatingPhysician', y='Count', order=operating_phy.index) #newly added


plt.xticks(size = 10) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of OperatingPhysician', fontsize=20)
plt.xlabel('OperatingPhysician', size = 14)
plt.ylabel('Count', size = 14)
plt.show()

train_outpatient['OperatingPhysician'].value_counts()[:10]



# checking nan values 
train_outpatient['OtherPhysician'].isnull().any()
train_outpatient['OtherPhysician']=train_outpatient['OtherPhysician'].fillna(0)
# remvoing nan with 0 in test otherphysican
test_outpatient['OtherPhysician']=test_outpatient['OtherPhysician'].fillna(0)

test_outpatient['OtherPhysician'].value_counts()

"""### claim diagnose code"""

len(train_outpatient['ClmDiagnosisCode_1'].unique())
# there are 10355 unique claimDaignosiscode

## replacing nan values with zero
train_outpatient['ClmDiagnosisCode_1'].isnull().any()
train_outpatient['ClmDiagnosisCode_1']=train_outpatient['ClmDiagnosisCode_1'].fillna(0)
train_outpatient['ClmDiagnosisCode_2']=train_outpatient['ClmDiagnosisCode_2'].fillna(0)
train_outpatient['ClmDiagnosisCode_3']=train_outpatient['ClmDiagnosisCode_3'].fillna(0)
train_outpatient['ClmDiagnosisCode_4']=train_outpatient['ClmDiagnosisCode_4'].fillna(0)
train_outpatient['ClmDiagnosisCode_5']=train_outpatient['ClmDiagnosisCode_5'].fillna(0)
train_outpatient['ClmDiagnosisCode_6']=train_outpatient['ClmDiagnosisCode_6'].fillna(0)
train_outpatient['ClmDiagnosisCode_7']=train_outpatient['ClmDiagnosisCode_7'].fillna(0)
train_outpatient['ClmDiagnosisCode_8']=train_outpatient['ClmDiagnosisCode_8'].fillna(0)
train_outpatient['ClmDiagnosisCode_9']=train_outpatient['ClmDiagnosisCode_9'].fillna(0)
train_outpatient['ClmDiagnosisCode_10']=train_outpatient['ClmDiagnosisCode_10'].fillna(0)
train_outpatient['ClmDiagnosisCode_1'].isnull().any()
# replacing nan valuse in test data
test_outpatient['ClmDiagnosisCode_1']=test_outpatient['ClmDiagnosisCode_1'].fillna(0)
test_outpatient['ClmDiagnosisCode_2']=test_outpatient['ClmDiagnosisCode_2'].fillna(0)
test_outpatient['ClmDiagnosisCode_3']=test_outpatient['ClmDiagnosisCode_3'].fillna(0)
test_outpatient['ClmDiagnosisCode_4']=test_outpatient['ClmDiagnosisCode_4'].fillna(0)
test_outpatient['ClmDiagnosisCode_5']=test_outpatient['ClmDiagnosisCode_5'].fillna(0)
test_outpatient['ClmDiagnosisCode_6']=test_outpatient['ClmDiagnosisCode_6'].fillna(0)
test_outpatient['ClmDiagnosisCode_7']=test_outpatient['ClmDiagnosisCode_7'].fillna(0)
test_outpatient['ClmDiagnosisCode_8']=test_outpatient['ClmDiagnosisCode_8'].fillna(0)
test_outpatient['ClmDiagnosisCode_9']=test_outpatient['ClmDiagnosisCode_9'].fillna(0)
test_outpatient['ClmDiagnosisCode_10']=test_outpatient['ClmDiagnosisCode_10'].fillna(0)


train_outpatient['ClmProcedureCode_1'].isnull().any()
## remving nan valeus form the claimprocedure
train_outpatient['ClmProcedureCode_1']=train_outpatient['ClmProcedureCode_1'].fillna(0)
train_outpatient['ClmProcedureCode_2']=train_outpatient['ClmProcedureCode_2'].fillna(0)
train_outpatient['ClmProcedureCode_3']=train_outpatient['ClmProcedureCode_3'].fillna(0)
train_outpatient['ClmProcedureCode_4']=train_outpatient['ClmProcedureCode_4'].fillna(0)
train_outpatient['ClmProcedureCode_5']=train_outpatient['ClmProcedureCode_5'].fillna(0)
train_outpatient['ClmProcedureCode_6']=train_outpatient['ClmProcedureCode_6'].fillna(0)
## removing nan values from test data
test_outpatient['ClmProcedureCode_1']=test_outpatient['ClmProcedureCode_1'].fillna(0)
test_outpatient['ClmProcedureCode_2']=test_outpatient['ClmProcedureCode_2'].fillna(0)
test_outpatient['ClmProcedureCode_3']=test_outpatient['ClmProcedureCode_3'].fillna(0)
test_outpatient['ClmProcedureCode_4']=test_outpatient['ClmProcedureCode_4'].fillna(0)
test_outpatient['ClmProcedureCode_5']=test_outpatient['ClmProcedureCode_5'].fillna(0)
test_outpatient['ClmProcedureCode_6']=test_outpatient['ClmProcedureCode_6'].fillna(0)

"""### DeductibleAmtPaid"""

train_outpatient['DeductibleAmtPaid'].isnull().any()

plt.figure(figsize=(16,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.distplot(train_outpatient['DeductibleAmtPaid'])  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of DeductibleAmtPaid', fontsize=20)
plt.xlabel('DeductibleAmtPaid', size = 14)
plt.ylabel('Density', size = 14)
plt.show()

for i in range(0,101,25):
    print('persentle {0} is {1}'.format(i,np.percentile(train_outpatient["DeductibleAmtPaid"],i),))

for i in range(90,101,1):
    print('persentle {0} is {1}'.format(i,np.percentile(train_outpatient["DeductibleAmtPaid"],i),))

for i in np.arange(99,100.1,0.1):
    print('persentle {0} is {1}'.format(i,np.percentile(np.absolute(train_outpatient["DeductibleAmtPaid"]),i)))

plt.figure(figsize=(7,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.boxplot(y=train_outpatient['DeductibleAmtPaid']) # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of DeductibleAmtPaid', fontsize=20)
plt.xlabel('DeductibleAmtPaid', size = 14)
plt.ylabel('Percentile', size = 14)
plt.show()




train_outpatient['ClmAdmitDiagnosisCode'].isnull().any()
# repaling nan values with 0.
train_outpatient['ClmAdmitDiagnosisCode']=train_outpatient['ClmAdmitDiagnosisCode'].fillna(0)
# removing from test data
test_outpatient['ClmAdmitDiagnosisCode']=test_outpatient['ClmAdmitDiagnosisCode'].fillna(0)

## top 20 clmAdmitDiagnosisicode
train_outpatient['ClmAdmitDiagnosisCode'].value_counts()[:20].index

train_outpatient.isna().sum()

#https://datatofish.com/correlation-matrix-pandas/#:~:text=Steps%20to%20Create%20a%20Correlation%20Matrix%20using%20Pandas,above%20dataset%20in%20Python%3A%20import...%20Step%203%3A%20
plt.figure(figsize=(15,8))
corrMatrix = train_outpatient.corr()
hm = sns.heatmap(corrMatrix, annot = True)
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12)
plt.show()



for i in train_inpatient.columns:
    if i not in train_outpatient.columns:
        print(i)

"""### DiagnosisGroupCode"""

train_inpatient.DiagnosisGroupCode.isnull().any()

dig_count=train_inpatient['DiagnosisGroupCode'].value_counts()[:30]

# county distribution
plt.figure(figsize=(20,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 

#sns.barplot(dig_count.index,dig_count.values,order=dig_count.index) # name of the category(index)  

df = pd.DataFrame({'Category': dig_count.index, 'Count': dig_count.values})
sns.barplot(data=df, x='Category', y='Count', order=dig_count.index)


train_beneficiary.columns



train_beneficiary.loc[train_beneficiary.DOD.isna(),'Dead_or_Alive']=0
train_beneficiary.loc[train_beneficiary.DOD.notna(),'Dead_or_Alive']=1
# test data

test_beneficiary.loc[test_beneficiary.DOD.isna(),"Dead_or_Alive"]=0
test_beneficiary.loc[test_beneficiary.DOD.notna(),"Dead_or_Alive"]=1



train_beneficiary.DOD.unique()

#https://datatofish.com/strings-to-datetime-pandas/
train_beneficiary['DOB']=pd.to_datetime(train_beneficiary['DOB'],format='%Y-%m-%d')
train_beneficiary['DOD']=pd.to_datetime(train_beneficiary['DOD'],format='%Y-%m-%d')
# test
test_beneficiary['DOD']=pd.to_datetime(test_beneficiary['DOD'],format='%Y-%m-%d')
test_beneficiary['DOB']=pd.to_datetime(test_beneficiary['DOB'],format='%Y-%m-%d')

# subracting dod form dob to get the age accoring to it.
train_beneficiary['Age']= round((train_beneficiary['DOD']-train_beneficiary['DOB']).dt.days/365)
#testB
test_beneficiary['Age']=round((test_beneficiary['DOD']-test_beneficiary['DOB']).dt.days/365)

train_beneficiary.Age.unique()
test_beneficiary.Age.unique()

# fill the nan vlaues accorindig to the latest date presnt in DoD.
train_beneficiary['Age']=train_beneficiary['Age'].fillna(round((pd.to_datetime('2009-12-01',format='%Y-%m-%d')-train_beneficiary['DOB']).dt.days/365))

# test
test_beneficiary['Age']=test_beneficiary['Age'].fillna(round((pd.to_datetime('2009-12-01',format='%Y-%m-%d')-test_beneficiary['DOB']).dt.days/365))
# cheking
train_beneficiary['Age'].isna().any()
test_beneficiary['Age'].isna().any()

age_count=train_beneficiary.Age.value_counts()
plt.figure(figsize=(30,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 

df=pd.DataFrame({"Category" : age_count.index, "Count" : age_count.values})
sns.barplot(data=df,x="Category",y="Count",order=age_count.index) # name of the category(index)  


plt.xticks(size = 10) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of Age', fontsize=20)
plt.xlabel('Age', size = 14)
plt.ylabel('Count', size = 14)
plt.show()



train_beneficiary['Tolat_chronic_cond']=  (train_beneficiary['ChronicCond_Alzheimer'] + train_beneficiary['ChronicCond_Cancer'] +
                                          train_beneficiary['ChronicCond_Depression'] + train_beneficiary['ChronicCond_Diabetes'] +
                                          train_beneficiary['ChronicCond_Heartfailure'] + train_beneficiary['ChronicCond_IschemicHeart'] +
                                          train_beneficiary['ChronicCond_KidneyDisease'] + train_beneficiary['ChronicCond_ObstrPulmonary'] +
                                           train_beneficiary['ChronicCond_rheumatoidarthritis'] + train_beneficiary['ChronicCond_stroke']
                                          )
# test 
test_beneficiary['Tolat_chronic_cond']=  (test_beneficiary['ChronicCond_Alzheimer'] + test_beneficiary['ChronicCond_Cancer'] +
                                          test_beneficiary['ChronicCond_Depression'] + test_beneficiary['ChronicCond_Diabetes'] +
                                          test_beneficiary['ChronicCond_Heartfailure'] + test_beneficiary['ChronicCond_IschemicHeart'] +
                                          test_beneficiary['ChronicCond_KidneyDisease'] + test_beneficiary['ChronicCond_ObstrPulmonary'] +
                                           test_beneficiary['ChronicCond_rheumatoidarthritis'] + test_beneficiary['ChronicCond_stroke']
                                          )

plt.figure(figsize=(12,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.countplot(data=train_beneficiary,x='Tolat_chronic_cond') # conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution of total_chornic_cond', fontsize=20)
plt.xlabel('tolal_chronic_condi', size = 14)
plt.ylabel('Density', size = 14)
plt.show()



plt.figure(figsize=(30,20))
corrMatrix = train_beneficiary.corr(method='spearman')
hm = sns.heatmap(corrMatrix, annot = True)
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12)
plt.show()

train_inpatient.columns


#Traindata
train_inpatient['Admitted_or_Not']=1
# test_data
test_inpatient['Admitted_or_Not']=1

#Traindata
train_outpatient['Admitted_or_Not']=0
# test_data
test_outpatient['Admitted_or_Not']=0


print(train_inpatient.shape,train_outpatient.shape)
# print(test_inpatient.shape,test_outpatient.shape)

# getting commonn columns
comm_col=[]
for i in train_inpatient.columns:
    if i in train_outpatient.columns:
        comm_col.append(i)
        print(i)
len(comm_col)


IN_OUT_train=pd.merge(train_inpatient,train_outpatient,left_on=comm_col,right_on=comm_col,how='outer')
IN_OUT_test=pd.merge(test_inpatient,test_outpatient,how='outer')
IN_OUT_train.columns
IN_OUT_test.columns


IN_OUT_train['AdmissionDt'].isna().any()
# IN_OUT_train['DischargeDt'].isna().sum()
# IN_OUT_test['AdmissionDt_x'].isna().any()

IN_OUT_train['AdmissionDt']=pd.to_datetime(IN_OUT_train['AdmissionDt'],format='%Y-%m-%d')
IN_OUT_train['DischargeDt']=pd.to_datetime(IN_OUT_train['DischargeDt'],format='%Y-%m-%d')
#test
IN_OUT_test['AdmissionDt']=pd.to_datetime(IN_OUT_test['AdmissionDt'],format='%Y-%m-%d')
IN_OUT_test['DischargeDt']=pd.to_datetime(IN_OUT_test['DischargeDt'],format='%Y-%m-%d')

IN_OUT_train['Admitted_days']=round((IN_OUT_train['DischargeDt']-IN_OUT_train['AdmissionDt']).dt.days)
#test
IN_OUT_test['Admitted_days']=round((IN_OUT_test['DischargeDt']-IN_OUT_test['AdmissionDt']).dt.days)

#filling nan values with 1 because patient is admitted for mininmun day is 1.
IN_OUT_train['Admitted_days']=IN_OUT_train['Admitted_days'].fillna(1)
IN_OUT_test['Admitted_days']=IN_OUT_test['Admitted_days'].fillna(1)
IN_OUT_train['Admitted_days'].isna().any()


IN_OUT_train['ClaimStartDt'].isna().any()
IN_OUT_train['ClaimEndDt'].isna().any()

IN_OUT_train['ClaimStartDt']=pd.to_datetime(IN_OUT_train['ClaimStartDt'],format='%Y-%m-%d')
IN_OUT_train['ClaimEndDt']=pd.to_datetime(IN_OUT_train['ClaimEndDt'],format='%Y-%m-%d')
#test
IN_OUT_test['ClaimStartDt']=pd.to_datetime(IN_OUT_test['ClaimStartDt'],format='%Y-%m-%d')
IN_OUT_test['ClaimEndDt']=pd.to_datetime(IN_OUT_test['ClaimEndDt'],format='%Y-%m-%d')

IN_OUT_train['Claim_time']=round((IN_OUT_train['ClaimEndDt']-IN_OUT_train['ClaimStartDt']).dt.days)+1 # adding one becase atleast 1 day to process the claim
#test
IN_OUT_test['Claim_time']=round((IN_OUT_test['ClaimEndDt']-IN_OUT_test['ClaimStartDt']).dt.days)+1



IN_OUT_train['InscClaimAmtReimbursed'].isna().any()
IN_OUT_train['DeductibleAmtPaid'].isna().any()

IN_OUT_train['DeductibleAmtPaid']=IN_OUT_train['DeductibleAmtPaid'].fillna(0)
#test
IN_OUT_test['DeductibleAmtPaid']=IN_OUT_test['DeductibleAmtPaid'].fillna(0)

IN_OUT_train['Amount_get']=IN_OUT_train['InscClaimAmtReimbursed']-IN_OUT_train['DeductibleAmtPaid']
#test
IN_OUT_test['Amount_get']=IN_OUT_test['InscClaimAmtReimbursed']-IN_OUT_test['DeductibleAmtPaid']

train_inpatient.ClmDiagnosisCode_4

IN_OUT_train=IN_OUT_train.fillna(0)


plt.figure(figsize=(30,20))
corrMatrix = IN_OUT_train.corr(method='spearman')
hm = sns.heatmap(corrMatrix, annot = True)
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12)
plt.show()


in_out_bene_train=pd.merge(IN_OUT_train,train_beneficiary,on='BeneID',how='inner')
#test data
in_out_bene_test=pd.merge(IN_OUT_test,test_beneficiary,on="BeneID",how='inner')

in_out_bene_train.head()
print(in_out_bene_train.shape,in_out_bene_test.shape)


final_data_train=pd.merge(in_out_bene_train,train_label,on='Provider',how='inner')
final_data_test=pd.merge(in_out_bene_test,test_label,on='Provider',how='inner')
print(final_data_train.shape,final_data_test.shape)

final_data_test.columns


final_data_train['Total_ip_op_amount_reimb']=final_data_train['IPAnnualReimbursementAmt']+final_data_train['OPAnnualReimbursementAmt']
# test
final_data_test['Total_ip_op_amount_reimb']=final_data_test['IPAnnualReimbursementAmt']+final_data_test['OPAnnualReimbursementAmt']



final_data_train['total_ip_op_amount_deduct']=final_data_train['OPAnnualDeductibleAmt']+final_data_train['IPAnnualDeductibleAmt']
# test
final_data_test['total_ip_op_amount_deduct']=final_data_test['OPAnnualDeductibleAmt']+final_data_test['IPAnnualDeductibleAmt']

final_data_train.head()

#checking null valeus in all  columns
for i in final_data_train.columns:
    print(i, "=" , final_data_train[i].isna().any())



final_data_train=final_data_train.fillna(0)
# test
final_data_test=final_data_test.fillna(0)

print(final_data_train.isna().any().tolist())


## race belong to which class
plt.figure(figsize=(12,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.FacetGrid(final_data_train, hue="PotentialFraud", height=5) \
   .map(sns.countplot, "Race").add_legend()# conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Race_belong_to_potentialFraud', fontsize=20)
plt.xlabel('Race', size = 14)
plt.ylabel('Count', size = 14)
plt.show()


# distributionn of age.
plt.figure(figsize=(12,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.FacetGrid(final_data_train, hue="PotentialFraud",height=10) \
   .map(sns.distplot, "Age").add_legend()# conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Age_belong_to_potentialFraud', fontsize=20)
plt.xlabel('Age', size = 14)
plt.ylabel('denisty', size = 14)
plt.show()


sns.FacetGrid(final_data_train, hue="PotentialFraud",height=6) \
   .map(sns.distplot, "Tolat_chronic_cond").add_legend()# conting unique values  
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Total_chronic_condition_belong_to_potentialFraud', fontsize=20)
plt.xlabel('Total_chronic_condtion', size = 14)
plt.ylabel('denisty', size = 14)
plt.show()


# age vs no of admitted days.
plt.figure(figsize=(12,6))
sns.lineplot(x='Age',y='Admitted_days',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Age vs Admitted_days', fontsize=20)
plt.xlabel('Age', size = 14)
plt.ylabel('Admitted_days', size = 14)
plt.show()


# displot of InscClaimAmtReimbursed
plt.figure(figsize=(12,6))
sns.set_style('whitegrid')
sns.displot(data=final_data_train, x="InscClaimAmtReimbursed", hue="PotentialFraud", kind="kde")
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution_Of_InscClaimAmtReimbursed', fontsize=20)
plt.xlabel('InscClaimAmtReimbursed', size = 14)
plt.ylabel('denisty', size = 14)
plt.show()



#CDF_Of_InscClaimAmtReimbursed
plt.figure(figsize=(12,6))
sns.displot(data=final_data_train, x="InscClaimAmtReimbursed", hue="PotentialFraud", kind="ecdf")
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('CDF_Of_InscClaimAmtReimbursed', fontsize=20)
plt.xlabel('InscClaimAmtReimbursed', size = 14)
plt.ylabel('denisty', size = 14)
plt.show()

"""### Observations
1. The cdf of inscClamAmt Remibused is also overlaping.
"""

#Displot_Of_DeductibleAmtPaid
plt.figure(figsize=(12,6))
sns.set_style('whitegrid')
sns.displot(data=final_data_train, x="DeductibleAmtPaid", hue="PotentialFraud", kind="hist",kde=True)
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Distribution_Of_DeductibleAmtPaid', fontsize=20)
plt.xlabel('DeductibleAmtPaid', size = 14)
plt.ylabel('Count', size = 14)
plt.show()

sns.displot(data=final_data_train, x="DeductibleAmtPaid", hue="PotentialFraud", kind="ecdf")
plt.title('CDF_Of_DeductibleAmtPaid', fontsize=20)
plt.xlabel('DeductibleAmtPaid', size = 14)
plt.ylabel('Proportion', size = 14)


plt.figure(figsize=(12,6))
sns.scatterplot(x='InscClaimAmtReimbursed',y='DeductibleAmtPaid',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('InscClaimAmtReimbursed vs DeductibleAmtPaid', fontsize=20)
plt.xlabel('InscClaimAmtReimbursed', size = 14)
plt.ylabel('DeductibleAmtPaid', size = 14)
plt.show()


plt.figure(figsize=(12,6))
sns.lineplot(x='Claim_time',y='Admitted_days',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Claim_time vs Admitted_days', fontsize=20)
plt.xlabel('Claim_time', size = 14)
plt.ylabel('Admitted_days', size = 14)
plt.show()


plt.figure(figsize=(12,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.FacetGrid(final_data_train, hue="PotentialFraud",height=10) \
   .map(sns.distplot, "IPAnnualReimbursementAmt").add_legend()# conting unique values  

plt.title('Distribution_Of_IPAnnualReimbursementAmt', fontsize=20)
plt.xlabel('IPAnnualReimbursementAmt', size = 14)
plt.ylabel('density', size = 14)
plt.show()

# distrubution of IPAnnualDeductibleAmt
plt.figure(figsize=(12,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.FacetGrid(final_data_train, hue="PotentialFraud",height=10) \
   .map(sns.distplot, "IPAnnualDeductibleAmt").add_legend()# conting unique values  

plt.title('Distribution_Of_IPAnnualDeductibleAmt', fontsize=20)
plt.xlabel('IPAnnualDeductibleAmt', size = 14)
plt.ylabel('density', size = 14)
plt.show()

# scatter plot of IPAnnualDeductibleAmt and IPAnnualReimbursementAmt
plt.figure(figsize=(12,6))
sns.scatterplot(x='IPAnnualDeductibleAmt',y='IPAnnualReimbursementAmt',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('IPAnnualDeductibleAmt vs IPAnnualReimbursementAmt', fontsize=20)
plt.xlabel('IPAnnualDeductibleAmt', size = 14)
plt.ylabel('IPAnnualReimbursementAmt', size = 14)
plt.show()
# final_data_test.IPAnnualReimbursementAmt


# distribution of OPAnnualReimbursementAmt
plt.figure(figsize=(12,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.FacetGrid(final_data_train, hue="PotentialFraud",height=10) \
   .map(sns.distplot, "OPAnnualReimbursementAmt").add_legend()# conting unique values  

plt.title('Distribution_Of_OPAnnualReimbursementAmt', fontsize=20)
plt.xlabel('OPAnnualReimbursementAmt', size = 14)
plt.ylabel('density', size = 14)
plt.show()

"""### Observations
1. Both curvs are overlaping 
"""

# distribution of OPAnnualDeductibleAmt
plt.figure(figsize=(12,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.FacetGrid(final_data_train, hue="PotentialFraud",height=10) \
   .map(sns.distplot, "OPAnnualDeductibleAmt").add_legend()# conting unique values  

plt.title('Distribution_Of_OPAnnualDeductibleAmt', fontsize=20)
plt.xlabel('OPAnnualDeductibleAmt', size = 14)
plt.ylabel('density', size = 14)
plt.show()


plt.figure(figsize=(12,6))
sns.scatterplot(x='OPAnnualReimbursementAmt',y='OPAnnualDeductibleAmt',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('OPAnnualReimbursementAmt vs OPAnnualDeductibleAmt', fontsize=20)
plt.xlabel('OPAnnualReimbursementAmt', size = 14)
plt.ylabel('OPAnnualDeductibleAmt', size = 14)
plt.show()



#distribution of total inpaitent outpatient reimbursemnet amount
plt.figure(figsize=(12,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.FacetGrid(final_data_train, hue="PotentialFraud",height=10) \
   .map(sns.distplot, "Total_ip_op_amount_reimb").add_legend()# conting unique values  

plt.title('Distribution_Of_Total_ip_op_amount_reimb', fontsize=20)
plt.xlabel('Total_ip_op_amount_reimb', size = 14)
plt.ylabel('density', size = 14)
plt.show()



# distributon of tolatl inpatient outpatient deductable amount.
plt.figure(figsize=(12,6)) # hight and width of plot 
sns.set_style('whitegrid') # backgroud of plot 
sns.FacetGrid(final_data_train, hue="PotentialFraud",height=10) \
   .map(sns.distplot, "total_ip_op_amount_deduct").add_legend()# conting unique values  

plt.title('Distribution_Of_total_ip_op_amount_deduct', fontsize=20)
plt.xlabel('total_ip_op_amount_deduct', size = 14)
plt.ylabel('density', size = 14)
plt.show()


plt.figure(figsize=(12,6))
sns.scatterplot(x='total_ip_op_amount_deduct',y='Total_ip_op_amount_reimb',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('total_ip_op_amount_deduct vs Total_ip_op_amount_reimb', fontsize=20)
plt.xlabel('total_ip_op_amount_deduct', size = 14)
plt.ylabel('Total_ip_op_amount_reimb', size = 14)
plt.show()


# line plot between chronic condition vs cliam time
plt.figure(figsize=(12,6))
sns.lineplot(x='Claim_time',y='Tolat_chronic_cond',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Claim_time vs Tolat_chronic_cond', fontsize=20)
plt.xlabel('Claim_time', size = 14)
plt.ylabel('Tolat_chronic_cond', size = 14)
plt.show()


# line plot between Amount_get vs cliam time
plt.figure(figsize=(12,6))
sns.lineplot(x='Claim_time',y='Amount_get',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Claim_time vs Amount_get', fontsize=20)
plt.xlabel('Claim_time', size = 14)
plt.ylabel('Amount_get', size = 14)
plt.show()



plt.figure(figsize=(30,20))
corrMatrix = final_data_train.corr()
hm = sns.heatmap(corrMatrix, annot = True)


plt.figure(figsize=(12,6))
sns.lineplot(x='InscClaimAmtReimbursed',y='Amount_get',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('InscClaimAmtReimbursed vs Amount_get', fontsize=20)
plt.xlabel('InscClaimAmtReimbursed', size = 14)
plt.ylabel('Amount_get', size = 14)
plt.show()


plt.figure(figsize=(12,6))
sns.lineplot(x='Admitted_days',y='DeductibleAmtPaid',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Admitted_days vs DeductibleAmtPaid', fontsize=20)
plt.xlabel('Admitted_days', size = 14)
plt.ylabel('DeductibleAmtPaid', size = 14)
plt.show()



plt.figure(figsize=(12,6))
sns.scatterplot(x='Total_ip_op_amount_reimb',y='IPAnnualReimbursementAmt',data=final_data_train,hue='PotentialFraud')
plt.xticks(size = 12) # size of x axis indicators(yes/no)
plt.yticks(size = 12) 
plt.title('Total_ip_op_amount_reimb vs IPAnnualReimbursementAmt', fontsize=20)
plt.xlabel('Total_ip_op_amount_reimb', size = 14)
plt.ylabel('IPAnnualReimbursementAmt', size = 14)
plt.show()


# spliting data
y=final_data_train['PotentialFraud']
# # split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]]
X_train, X_cv, y_train, y_cv = train_test_split(final_data_train, y,stratify=y,test_size=0.33,random_state=42)
# print('Number of data points in train data:', X_train.shape)
# # print('Number of data points in test data:', X_test.shape[0])
# print('Number of data points in cross validation data:', X_cv.shape)



# # we are getting mean of provider on x_train and X_test seprately to avoid data leakage proplem

mean_df=X_train[["InscClaimAmtReimbursed",'Provider']].groupby('Provider') # grouping the InsclaimAmtReimbursed on Provider
mean=mean_df.aggregate(np.mean) # getting mean of each group
provider_id=mean_df.groups # getting group names
g=list(provider_id.keys())
# adding mean of that group with the provider in new column 
from tqdm import tqdm   
for i,j in tqdm(zip(g,mean['InscClaimAmtReimbursed'])):
    X_train.loc[X_train['Provider'] == i, 'Mean_InscClaimAmtReimbursed'] = j

mean_df=X_cv[["InscClaimAmtReimbursed",'Provider']].groupby('Provider')
mean=mean_df.aggregate(np.mean)
provider_id=mean_df.groups
g=list(provider_id.keys())
from tqdm import tqdm   
for i,j in tqdm(zip(g,mean['InscClaimAmtReimbursed'])):
    X_cv.loc[X_cv['Provider'] == i, 'Mean_InscClaimAmtReimbursed'] = j

"""###### IPAnnualReimbursementAmt"""

mean_df=X_train[["IPAnnualReimbursementAmt",'Provider']].groupby('Provider')
mean=mean_df.aggregate(np.mean)
provider_id=mean_df.groups
g=list(provider_id.keys())

from tqdm import tqdm   
for i,j in tqdm(zip(g,mean['IPAnnualReimbursementAmt'])):
    X_train.loc[X_train['Provider'] == i, 'Mean_IPAnnualReimbursementAmt'] = j

mean_df=X_cv[["IPAnnualReimbursementAmt",'Provider']].groupby('Provider')
mean=mean_df.aggregate(np.mean)
provider_id=mean_df.groups
g=list(provider_id.keys())
from tqdm import tqdm   
for i,j in tqdm(zip(g,mean['IPAnnualReimbursementAmt'])):
    X_cv.loc[X_cv['Provider'] == i, 'Mean_IPAnnualReimbursementAmt'] = j

"""###### OPAnnualReimbursementAmt"""

mean_df=X_train[["OPAnnualReimbursementAmt",'Provider']].groupby('Provider')
mean=mean_df.aggregate(np.mean)
provider_id=mean_df.groups
g=list(provider_id.keys())

from tqdm import tqdm   
for i,j in tqdm(zip(g,mean['OPAnnualReimbursementAmt'])):
    X_train.loc[X_train['Provider'] == i, 'Mean_OPAnnualReimbursementAmt'] = j

mean_df=X_cv[["OPAnnualReimbursementAmt",'Provider']].groupby('Provider')
mean=mean_df.aggregate(np.mean)
provider_id=mean_df.groups
g=list(provider_id.keys())
from tqdm import tqdm   
for i,j in tqdm(zip(g,mean['OPAnnualReimbursementAmt'])):
    X_cv.loc[X_cv['Provider'] == i, 'Mean_OPAnnualReimbursementAmt'] = j

"""#### 2.Count featues

##### Getting Count of diffrent physician Attended a beneficiary
"""

# train_data
df_physician=X_train[['AttendingPhysician','OperatingPhysician','OtherPhysician']] # creating a new dataframe
c = np.where(df_physician==0,0,1) # replacing the physican code with 1
sum_total_featues=np.sum(c,axis=1) # adding all columns.

X_train['Total_physican_attended']=sum_total_featues # stroing in new column

# test
df_physician=X_cv[['AttendingPhysician','OperatingPhysician','OtherPhysician']]
c = np.where(df_physician==0,0,1)
sum_total_featues=np.sum(c,axis=1)

X_cv['Total_physican_attended']=sum_total_featues

"""##### #Getting Count of diffrent ClmDiagnosisCode"""

#train
li=[] # ceateing list of all ClmDiagonosisCode
for i in range(1,11):
    li.append("ClmDiagnosisCode_"+str(i))    
df_=X_train[li] # storing in a diffrent dataframe
c = np.where(df_==0,0,1) # Changing the code with 1 and 0.
sum_total_featues=np.sum(c,axis=1) # summing all 1's column vise.
X_train['Total_ClmDiagnosisCode']=sum_total_featues # storing in diffrent column

# test
li=[]
for i in range(1,11):
    li.append("ClmDiagnosisCode_"+str(i))    
df_=X_cv[li]             # storing in a diffrent dataframe
c = np.where(df_==0,0,1)  # Changing the code with 1 and 0.
sum_total_featues=np.sum(c,axis=1) 
X_cv['Total_ClmDiagnosisCode']=sum_total_featues

"""###### Getting Count of diffrent ClmProcedureCode"""

#train
li=[]  # ceateing list of all ClmProcedureCode
for i in range(1,7):
    li.append("ClmProcedureCode_"+str(i))    
df_=X_train[li]    # storing in a diffrent dataframe
c = np.where(df_==0,0,1)   # Changing the code with 1 and 0.
sum_total_featues=np.sum(c,axis=1)    # summing all 1's column vise
X_train['Total_ClmProcedureCode']=sum_total_featues

#test
li=[]
for i in range(1,7):
    li.append("ClmProcedureCode_"+str(i))    
df_=X_cv[li]
c = np.where(df_==0,0,1)
sum_total_featues=np.sum(c,axis=1)
X_cv['Total_ClmProcedureCode']=sum_total_featues


# # creating dataframe which only contains fraud data.
fraud_data=X_train[X_train['PotentialFraud']==1] # from trian data only.


claim_adimt_code_train=fraud_data['ClmAdmitDiagnosisCode'].value_counts()[:21] # only top 20 code exculding 0/nan values
# changing codes in trian data if they claim_admit code present in the column or not if not replace it by none
top_20_claim_admit_code=claim_adimt_code_train.keys()
claim_code_allowed_train=list(top_20_claim_admit_code)
print(claim_code_allowed_train)

if 0 in claim_code_allowed_train:  # Removing zero form the list. 
    claim_code_allowed_train.remove(0)
for i in claim_code_allowed_train:
#     print(i)
# adding new featues with that category to the  train data.
# replacing that code with 1 and other with 0. 
    X_train['ClmAdmitDiagnosisCode_'+i]=np.where(X_train["ClmAdmitDiagnosisCode"].str.contains(i), 1, 0)

claim_adimt_code_train=fraud_data['ClmAdmitDiagnosisCode'].value_counts()[:21] # only top 20 code exculding 0/nan values
# changing codes in trian data if they claim_admit code present in the column or not if not replace it by none
top_20_claim_admit_code=claim_adimt_code_train.keys()
claim_code_allowed_train=list(top_20_claim_admit_code)
print(claim_code_allowed_train)

if 0 in claim_code_allowed_train:
    claim_code_allowed_train.remove(0)
for i in claim_code_allowed_train:
#     print(i)
    X_cv['ClmAdmitDiagnosisCode_'+i]=np.where(X_cv["ClmAdmitDiagnosisCode"].str.contains(i), 1, 0)


for i in range(1,11):
    code=fraud_data['ClmDiagnosisCode_'+str(i)].value_counts()[:10]  # craetind list of code for each ClmDiagnosisCode column
    code=code.keys()
    code=list(code)
    if 0 in code: # removing code zero. if occure
        code.remove(0)
    for k in code:

        X_train['ClmDiagnosisCode_'+k]=np.where(X_train["ClmDiagnosisCode_"+str(i)].str.contains(k), 1, 0) # replacing the code with 1. and other values with zero.

for i in range(1,11):
    code=fraud_data['ClmDiagnosisCode_'+str(i)].value_counts()[:10]
    code=code.keys()
    code=list(code)
    if 0 in code:
        code.remove(0)
    for k in code:

        X_cv['ClmDiagnosisCode_'+k]=np.where(X_cv["ClmDiagnosisCode_"+str(i)].str.contains(k), 1, 0)
        
print(X_train.shape,X_cv.shape)


claim_adimt_code_train=fraud_data['AttendingPhysician'].value_counts()[:10] # only top 20 code exculding 0/nan values
# changing codes in trian data if they claim_admit code present in the column or not if not replace it by none
top_20_claim_admit_code=claim_adimt_code_train.keys()
claim_code_allowed_train=list(top_20_claim_admit_code)
print(claim_code_allowed_train)
if 0 in claim_code_allowed_train:
    
    claim_code_allowed_train.remove(0)
for i in claim_code_allowed_train:
#     print(i)
# replacing the code with 1. and other values with zero.
    X_train['AttendingPhysician_'+i]=np.where(X_train["AttendingPhysician"].str.contains(i), 1, 0)

claim_adimt_code_train=fraud_data['AttendingPhysician'].value_counts()[:10] # only top 20 code exculding 0/nan values
# changing codes in trian data if they claim_admit code present in the column or not if not replace it by none
top_20_claim_admit_code=claim_adimt_code_train.keys()
claim_code_allowed_train=list(top_20_claim_admit_code)
print(claim_code_allowed_train)
if 0 in claim_code_allowed_train:
    
    claim_code_allowed_train.remove(0)
for i in claim_code_allowed_train:
#     print(i)
    X_cv['AttendingPhysician_'+i]=np.where(X_cv["AttendingPhysician"].str.contains(i), 1, 0)
print(X_train.shape,X_cv.shape)


claim_adimt_code_train=fraud_data['OperatingPhysician'].value_counts()[:10] # only top 20 code exculding 0/nan values
# changing codes in trian data if they claim_admit code present in the column or not if not replace it by none
top_20_claim_admit_code=claim_adimt_code_train.keys()
claim_code_allowed_train=list(top_20_claim_admit_code)
print(claim_code_allowed_train)
if 0 in claim_code_allowed_train:
    claim_code_allowed_train.remove(0)
for i in claim_code_allowed_train:
    print(i)
    X_train['OperatingPhysician_'+i]=np.where(X_train["OperatingPhysician"].str.contains(i), 1, 0)


print(X_train.shape,X_cv.shape)



claim_adimt_code_train=fraud_data['OperatingPhysician'].value_counts()[:10] # only top 20 code exculding 0/nan values
# changing codes in trian data if they claim_admit code present in the column or not if not replace it by none
top_20_claim_admit_code=claim_adimt_code_train.keys()
claim_code_allowed_train=list(top_20_claim_admit_code)
print(claim_code_allowed_train)
if 0 in claim_code_allowed_train:
    claim_code_allowed_train.remove(0)
for i in claim_code_allowed_train:
    print(i)
    X_cv['OperatingPhysician_'+i]=np.where(X_cv["OperatingPhysician"].str.contains(i), 1, 0)


print(X_train.shape,X_cv.shape)


claim_adimt_code_train=fraud_data['OtherPhysician'].value_counts()[:10] # only top 20 code exculding 0/nan values
# changing codes in trian data if they claim_admit code present in the column or not if not replace it by none
top_20_claim_admit_code=claim_adimt_code_train.keys()
claim_code_allowed_train=list(top_20_claim_admit_code)
print(claim_code_allowed_train)
if 0 in claim_code_allowed_train:
    claim_code_allowed_train.remove(0)
for i in claim_code_allowed_train:
    print(i)
    X_train['OtherPhysician_'+i]=np.where(X_train["OtherPhysician"].str.contains(i), 1, 0)

claim_adimt_code_train=fraud_data['OtherPhysician'].value_counts()[:10] # only top 20 code exculding 0/nan values
# changing codes in trian data if they claim_admit code present in the column or not if not replace it by none
top_20_claim_admit_code=claim_adimt_code_train.keys()
claim_code_allowed_train=list(top_20_claim_admit_code)
print(claim_code_allowed_train)
if 0 in claim_code_allowed_train:
    claim_code_allowed_train.remove(0)
for i in claim_code_allowed_train:
    print(i)
    X_cv['OtherPhysician_'+i]=np.where(X_cv["OtherPhysician"].str.contains(i), 1, 0)
    
print(X_train.shape,X_cv.shape)


# fraud_data=X_train[X_train['PotentialFraud']==1]
print('Maximum IPAnnualreimbursementAmt:-',fraud_data['IPAnnualReimbursementAmt'].max())
print( 'Maximum OPAnnualReimbursementAmt:-',fraud_data['OPAnnualReimbursementAmt'].max())
print('Maximum InscClaimAmtReimbursed:-',fraud_data['InscClaimAmtReimbursed'].max())

#Diff_max_IPAnnualReimbursementAm
X_train['Diff_max_IPAnnualReimbursementAmt']=fraud_data['IPAnnualReimbursementAmt'].max()-X_train['IPAnnualReimbursementAmt']
# OPAnnualReimbursementAmt
X_train['Diff_max_OPAnnualReimbursementAmt']=fraud_data['OPAnnualReimbursementAmt'].max()-X_train['OPAnnualReimbursementAmt']
#InscClaimAmtReimbursed
X_train['Diff_max_InscClaimAmtReimbursed']=fraud_data['InscClaimAmtReimbursed'].max()-X_train['InscClaimAmtReimbursed']

print(X_train.shape,X_cv.shape)

# #deviation
# fraud_data=X_cv[X_cv['PotentialFraud']==1]
print('Maximum IPAnnualreimbursementAmt:-',fraud_data['IPAnnualReimbursementAmt'].max())
print( 'Maximum OPAnnualReimbursementAmt:-',fraud_data['OPAnnualReimbursementAmt'].max())

#IPAnnualReimbursementAmt
X_cv['Diff_max_IPAnnualReimbursementAmt']=fraud_data['IPAnnualReimbursementAmt'].max()-X_cv['IPAnnualReimbursementAmt']
# OpannualReimbursemetamt
X_cv['Diff_max_OPAnnualReimbursementAmt']=fraud_data['OPAnnualReimbursementAmt'].max()-X_cv['OPAnnualReimbursementAmt']
#InscClaimAmtReimbursed
X_cv['Diff_max_InscClaimAmtReimbursed']=fraud_data['InscClaimAmtReimbursed'].max()-X_cv['InscClaimAmtReimbursed']

print(X_train.shape,X_cv.shape)


# for country one hot encoding.
contry_code=X_train['County'].unique()
# contry_code=map(str,contry_code)
for i in contry_code:
    X_train['County_'+str(i)]=np.where(X_train["County"]==i, 1, 0)
# test data

for i in contry_code:
    
    X_cv['County_'+str(i)]=np.where(X_cv["County"]==i, 1, 0)

# Convert type of Gender and Race to categorical
X_train.Gender=X_train.Gender.astype('category')
X_cv.Gender=X_cv.Gender.astype('category')

X_train.Race=X_train.Race.astype('category')
X_cv.Race=X_cv.Race.astype('category')

X_train.State=X_train.State.astype('category')
# X_train.County=X_train.County.astype('category')

X_cv.State=X_cv.State.astype('category')
# X_cv.County=X_cv.County.astype('category')

X_train.NoOfMonths_PartACov=X_train.NoOfMonths_PartACov.astype('category')
X_train.NoOfMonths_PartBCov=X_train.NoOfMonths_PartBCov.astype('category')

X_cv.NoOfMonths_PartACov=X_cv.NoOfMonths_PartACov.astype('category')
X_cv.NoOfMonths_PartBCov=X_cv.NoOfMonths_PartBCov.astype('category')
# Do one hot encoding for gender and Race
X_train=pd.get_dummies(X_train,columns=['Gender','Race','State','NoOfMonths_PartBCov','NoOfMonths_PartACov'])

X_cv=pd.get_dummies(X_cv,columns=['Gender','Race','State','NoOfMonths_PartBCov','NoOfMonths_PartACov'])

print(X_train.shape,X_cv.shape)


remove_col=['AttendingPhysician','OperatingPhysician','OtherPhysician','ClmAdmitDiagnosisCode','DiagnosisGroupCode',
           'Provider','BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt',"DOD",'DOB','AdmissionDt','County',
           'DischargeDt','Provider','County']
for i in range(1,11):
    remove_col.append('ClmDiagnosisCode_'+str(i))
for i in range(1,7):
    remove_col.append('ClmProcedureCode_'+str(i))
    
    
print(remove_col)

X_train=X_train.drop(columns=remove_col,axis=1)
X_cv=X_cv.drop(columns=remove_col,axis=1) 
print(X_train.shape,X_cv.shape)


print(X_train.shape,X_cv.shape)


# removing values greater then 99.8 percentile
list_1=['IPAnnualDeductibleAmt','IPAnnualDeductibleAmt','OPAnnualDeductibleAmt','OPAnnualReimbursementAmt',
 'InscClaimAmtReimbursed','Total_ip_op_amount_reimb','total_ip_op_amount_deduct']
for i in list_1:
    X_train_w=X_train[X_train[i]<np.percentile(X_train[i],99.8)]
    X_cv_w=X_cv[X_cv[i]<np.percentile(X_cv[i],99.8)]

# # remvoing collinear featues

col=['Admitted_days','Amount_get','IPAnnualReimbursementAmt']
X_train_w=X_train_w.drop(columns=col,axis=1)
X_cv_w=X_cv_w.drop(columns=col,axis=1)

y_train_w=X_train_w['PotentialFraud']
y_cv_w=X_cv_w['PotentialFraud']
X_train=X_train.drop(columns=['PotentialFraud'],axis=1)
X_cv=X_cv.drop(columns=['PotentialFraud'],axis=1)


# saving in csv_files
X_train_w=X_train_w.drop(columns=['PotentialFraud'],axis=1)
X_cv_w=X_cv_w.drop(columns=['PotentialFraud'],axis=1) 

X_train_w.to_csv('X_train_w.csv')
X_cv_w.to_csv('X_cv_w.csv')
y_train_w.to_csv('y_train_w.csv')
y_cv_w.to_csv('y_cv_w.csv')
print(X_train_w.shape,X_cv_w.shape)


X_train.to_csv('X_train.csv')
y_train.to_csv('y_train.csv')
X_cv.to_csv('X_cv.csv')
y_cv.to_csv('y_cv.csv')







