from flask import Flask, render_template
from flask import Flask, jsonify, request
import numpy as np
import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import lightgbm as lgb
import gunicorn
import os
import warnings

app = Flask(__name__)



def final_pipeline(train_beneficiary,train_inpatient,train_outpatient):
    list_obj=['AttendingPhysician','OperatingPhysician','OtherPhysician']
    for i in list_obj:
        # print(i)
        train_inpatient[i]=train_inpatient[i].astype(object)
        train_outpatient[i]=train_outpatient[i].astype(object)
    for i in range(1,11):
        j="ClmDiagnosisCode_"+str(i)
        train_inpatient[j]=train_inpatient[j].astype(str)
        train_outpatient[j]=train_outpatient[j].astype(str)
    for i in range(1,7):
        j="ClmProcedureCode_"+str(i)
        train_inpatient[j]=train_inpatient[j].astype(str)
        train_outpatient[j]=train_outpatient[j].astype(str)    
    
    print('data loaded')
    train_beneficiary['Gender']=train_beneficiary['Gender'].replace(2,0)    
    train_beneficiary['RenalDiseaseIndicator']=train_beneficiary['RenalDiseaseIndicator'].replace('Y',int(1))
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
    train_beneficiary.loc[train_beneficiary.DOD.isna(),'Dead_or_Alive']=0
    train_beneficiary.loc[train_beneficiary.DOD.notna(),'Dead_or_Alive']=1
    train_beneficiary['DOB']=pd.to_datetime(train_beneficiary['DOB'],format='%Y-%m-%d')
    train_beneficiary['DOD']=pd.to_datetime(train_beneficiary['DOD'],format='%Y-%m-%d')
    train_beneficiary['Age']= round((train_beneficiary['DOD']-train_beneficiary['DOB']).dt.days/365)
    train_beneficiary['Age']=train_beneficiary['Age'].fillna(round((pd.to_datetime('2009-12-01',format='%Y-%m-%d')-train_beneficiary['DOB']).dt.days/365))
    train_inpatient['Admitted_or_Not']=1
    train_outpatient['Admitted_or_Not']=0
    
    comm_col=[]
    for i in train_inpatient.columns:
        if i in train_outpatient.columns:
            comm_col.append(i)
    len(comm_col)
  
    IN_OUT_train=pd.merge(train_inpatient,train_outpatient,left_on=comm_col,right_on=comm_col,how='outer')  
    IN_OUT_train['AdmissionDt']=pd.to_datetime(IN_OUT_train['AdmissionDt'],format='%Y-%m-%d')
    IN_OUT_train['DischargeDt']=pd.to_datetime(IN_OUT_train['DischargeDt'],format='%Y-%m-%d')  
    IN_OUT_train['Admitted_days']=round((IN_OUT_train['DischargeDt']-IN_OUT_train['AdmissionDt']).dt.days)
    IN_OUT_train['Admitted_days']=IN_OUT_train['Admitted_days'].fillna(1)
    IN_OUT_train['ClaimStartDt']=pd.to_datetime(IN_OUT_train['ClaimStartDt'],format='%Y-%m-%d')
    IN_OUT_train['ClaimEndDt']=pd.to_datetime(IN_OUT_train['ClaimEndDt'],format='%Y-%m-%d')
    IN_OUT_train['Claim_time']=round((IN_OUT_train['ClaimEndDt']-IN_OUT_train['ClaimStartDt']).dt.days)+1     
    final_data_train=pd.merge(IN_OUT_train,train_beneficiary,on='BeneID',how='inner')
    print('removing Nan values')
    final_data_train=final_data_train.fillna(0)
    final_data_train['Tolat_chronic_cond']=  (final_data_train['ChronicCond_Alzheimer'] + final_data_train['ChronicCond_Cancer'] +
                                          final_data_train['ChronicCond_Depression'] + final_data_train['ChronicCond_Diabetes'] +
                                          final_data_train['ChronicCond_Heartfailure'] + final_data_train['ChronicCond_IschemicHeart'] +
                                          final_data_train['ChronicCond_KidneyDisease'] + final_data_train['ChronicCond_ObstrPulmonary'] +
                                           final_data_train['ChronicCond_rheumatoidarthritis'] + final_data_train['ChronicCond_stroke'])
                                              
    final_data_train['Amount_get']=final_data_train['InscClaimAmtReimbursed']-final_data_train['DeductibleAmtPaid']
    final_data_train['Total_ip_op_amount_reimb']=final_data_train['IPAnnualReimbursementAmt']+final_data_train['OPAnnualReimbursementAmt']  
    final_data_train['total_ip_op_amount_deduct']=final_data_train['OPAnnualDeductibleAmt']+final_data_train['IPAnnualDeductibleAmt']
    X_train=final_data_train
    df_physician=X_train[['AttendingPhysician','OperatingPhysician','OtherPhysician']] 
    c = np.where(df_physician==0,0,1) 
    sum_total_featues=np.sum(c,axis=1) 
    X_train['Total_physican_attended']=sum_total_featues 
    
    li=[] 
    for i in range(1,11):
        li.append("ClmDiagnosisCode_"+str(i))    
    df_=X_train[li] 
    c = np.where(df_==0,0,1) 
    sum_total_featues=np.sum(c,axis=1)
    X_train['Total_ClmDiagnosisCode']=sum_total_featues 
    
    li=[]  
    for i in range(1,7):
        li.append("ClmProcedureCode_"+str(i))    
    df_=X_train[li]    
    c = np.where(df_==0,0,1)   
    sum_total_featues=np.sum(c,axis=1)    
    X_train['Total_ClmProcedureCode']=sum_total_featues
    
    claim_diagnosis_code={'ClmDiagnosisCode_1': ['4019', '4011', '2724', '42731', '2720', '2722', '2721', '2723', '78659'], 
                      'ClmDiagnosisCode_2': ['4019', '25000', '2724', 'V5861', 'V5869', '42731', '2449', '2720', '4280'], 
                      'ClmDiagnosisCode_3': ['4019', '25000', '2724', 'V5869', 'V5861', '2449', '42731', '2720', '4280'], 
                      'ClmDiagnosisCode_4': ['4019', '25000', '2724', 'V5869', '42731', '2449', '53081', '2720', '4280'], 
                      'ClmDiagnosisCode_5': ['4019', '25000', '2724', '42731', '53081', '2449', 'V5869', '41401', '4280'], 
                      'ClmDiagnosisCode_6': ['4019', '25000', '2724', '4280', '53081', '41401', '42731', '2449', '496'], 
                      'ClmDiagnosisCode_7': ['4019', '25000', '2724', '4280', '42731', '41401', '53081', '2449', '496'], 
                      'ClmDiagnosisCode_8': ['4019', '25000', '2724', '4280', '42731', '41401', '53081', '496', '5990'], 
                      'ClmDiagnosisCode_9': ['4019', '25000', '2724', '42731', '41401', '4280', '53081', '5990', '496'], 
                      'ClmDiagnosisCode_10': ['4019', '2724', '25000', '53081', '5990', '4280', '42731', '41401', '2449']}


    clmAdmitDiagnosisCode=[0, '42731', 'V7612', '78605', '78650', '78900', '4019', '25000', '486', '78079',
                           '7802', '7295', '5990', 'V5883', '4280', '7242', '7862', 'V5789', '2724', 'V5861', '78097']

    Attendig_physicain=['PHY330576', 'PHY350277', 'PHY412132', 'PHY423534', 'PHY314027', 'PHY357120', 
                        'PHY337425', 'PHY338032', 'PHY341578', 'PHY327046']

    Operating_physician= [0, 'PHY330576', 'PHY424897', 'PHY357120', 'PHY314027', 'PHY333735', 'PHY412132',
                          'PHY423534', 'PHY381249', 'PHY337425']

    Other_physician= [0, 'PHY412132', 'PHY341578', 'PHY338032', 'PHY337425', 'PHY347064', 
                      'PHY322092', 'PHY409965', 'PHY313818', 'PHY350277']

    Maximum_IPAnnualreimbursementAmt= 161470
    Maximum_OPAnnualReimbursementAmt=102960
    Maximum_InscClaimAmtReimbursed= 125000

    contry_code =[240, 411, 930, 970, 892, 380, 390, 410, 720, 590, 670, 620, 300, 0, 750, 90, 10, 580, 490, 70, 200, 770, 921, 80,
                  700, 989, 310, 988, 711, 141, 810, 140, 510, 150, 210, 20, 550, 920, 110, 420, 622, 250, 40, 340, 360, 430, 60, 
                  940, 790, 551, 600, 540, 890, 843, 480, 120, 230, 560, 260, 948, 860, 100, 330, 341, 400, 320, 190, 830, 520, 
                  160, 280, 910, 130, 370, 760, 50, 180, 974, 470, 290, 650, 780, 500, 984, 270, 751, 842, 680, 570, 610, 640, 30, 
                  460, 450, 170, 977, 999, 350, 630, 820, 440, 880, 850, 220, 946, 861, 331, 882, 311, 870, 541, 530, 991, 801, 
                  840, 980, 832, 660, 710, 802, 800, 662, 740, 981, 792, 621, 251, 564, 730, 950, 881, 291, 953, 194, 971, 960, 
                  986, 451, 791, 812, 990, 113, 562, 867, 794, 690, 913, 111, 757, 342, 982, 783, 885, 954, 381, 758, 821, 838, 
                  641, 973, 1, 993, 611, 891, 561, 321, 947, 213, 511, 902, 671, 591, 883, 961, 581, 886, 756, 978, 11, 900, 288, 
                  752, 871, 761, 887, 421, 975, 772, 845, 994, 911, 722, 851, 785, 241, 653, 945, 224, 901, 731, 983, 793, 976, 
                  563, 312, 942, 831, 34, 979, 462, 841, 943, 835, 542, 461, 992, 703, 471, 88, 301, 905, 795, 661, 734, 944, 955,
                  281, 681, 962, 904, 701, 985, 531, 811, 888, 552, 191, 743, 161, 951, 25, 879, 874, 651, 392, 222, 84, 691, 654, 
                  754, 672, 771, 702, 844, 362, 862, 361, 784, 631, 441, 952, 987, 292, 55, 884, 343, 211, 941, 996, 601, 822, 131,
                  782, 893, 592, 875, 755, 582, 431, 797, 652, 221, 741, 972, 223, 878, 391, 271, 803, 932, 117, 212, 949, 14, 522,
                  328, 632, 876, 412, 903, 963, 796, 912, 583, 744, 873, 712, 931, 753, 612]

    gender_code=[0, 1]
    race_code=[1, 2, 5, 3]
    state_code= [33, 49, 44, 6, 45, 5, 28, 34, 23, 16, 36, 51, 3, 52, 26, 22, 21, 11, 19, 14, 31, 46, 4, 10, 20, 12, 18, 37, 50, 
                25, 29, 39, 1, 38, 15, 13, 42, 17, 54, 35, 32, 30, 8, 47, 24, 7, 43, 53, 27, 9, 2, 41]

    NoOfMonths_PartACov_code= [12, 0, 11, 10, 6, 9, 7, 8, 4, 3, 5, 1, 2]

    NoOfMonths_PartBCov_code= [12, 5, 0, 6, 7, 1, 10, 11, 8, 9, 3, 2, 4]

    if 0 in clmAdmitDiagnosisCode:  # Removing zero form the list. 
        clmAdmitDiagnosisCode.remove(0)
    X_train["ClmAdmitDiagnosisCode"]=str(X_train["ClmAdmitDiagnosisCode"])
    for i in clmAdmitDiagnosisCode:
        X_train['ClmAdmitDiagnosisCode_'+i]=np.where(X_train["ClmAdmitDiagnosisCode"].str.contains(i), 1, 0) 
        
    for i in range(1,11):

        code=claim_diagnosis_code.get('ClmDiagnosisCode_'+str(i))  # craetind list of code for each ClmDiagnosisCode column

        if 0 in code: # removing code zero. if occure
            code.remove(0)
        X_train["ClmDiagnosisCode_"+str(i)]=str(X_train["ClmDiagnosisCode_"+str(i)])
        for k in code:
            X_train['ClmDiagnosisCode_'+k]=np.where(X_train["ClmDiagnosisCode_"+str(i)].str.contains(k), 1, 0) # replacing the code with 1. and other values with zero.
            
    if 0 in Attendig_physicain:
        Attendig_physicain.remove(0)
    X_train["AttendingPhysician"]=str(X_train["AttendingPhysician"])
    for i in Attendig_physicain:    
        X_train['AttendingPhysician_'+i]=np.where(X_train["AttendingPhysician"].str.contains(i), 1, 0)
    if 0 in Operating_physician:
        Operating_physician.remove(0)
    X_train["OperatingPhysician"]=str(X_train["OperatingPhysician"])
    for i in Operating_physician:
        X_train["AttendingPhysician"]=str(X_train["AttendingPhysician"])
        X_train['OperatingPhysician_'+i]=np.where(X_train["OperatingPhysician"].str.contains(i), 1, 0)


    print(X_train.shape)
    if 0 in Other_physician:
        Other_physician.remove(0)
    X_train["OtherPhysician"]=str(X_train["OtherPhysician"])
    for i in Other_physician:
        X_train['OtherPhysician_'+i]=np.where(X_train["OtherPhysician"].str.contains(i), 1, 0)
        
    X_train['Diff_max_IPAnnualReimbursementAmt']=Maximum_IPAnnualreimbursementAmt-X_train['IPAnnualReimbursementAmt']
    X_train['Diff_max_OPAnnualReimbursementAmt']=Maximum_OPAnnualReimbursementAmt-X_train['OPAnnualReimbursementAmt']
    X_train['Diff_max_InscClaimAmtReimbursed']=Maximum_InscClaimAmtReimbursed-X_train['InscClaimAmtReimbursed']
    print(X_train.shape)
    for i in contry_code:
        X_train['County_'+str(i)]=np.where(X_train["County"]==i, 1, 0)
    for i in gender_code:
        X_train['Gender_'+str(i)]=np.where(X_train["Gender"]==i, 1, 0)
    for i in race_code:
        X_train['Race_'+str(i)]=np.where(X_train["Race"]==i, 1, 0)        
    for i in state_code:
        X_train['State_'+str(i)]=np.where(X_train["State"]==i, 1, 0)
    for i in NoOfMonths_PartACov_code:
        X_train['NoOfMonths_PartACov_'+str(i)]=np.where(X_train["NoOfMonths_PartACov"]==i, 1, 0) 
    for i in NoOfMonths_PartBCov_code:
        X_train['NoOfMonths_PartBCov_'+str(i)]=np.where(X_train["NoOfMonths_PartBCov"]==i, 1, 0) 
    providr_id=X_train['Provider']    
    remove_col=['AttendingPhysician','OperatingPhysician','OtherPhysician','ClmAdmitDiagnosisCode','DiagnosisGroupCode',
           'Provider','BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt',"DOD",'DOB','AdmissionDt','Gender',
               'DischargeDt','Provider','County','NoOfMonths_PartACov','NoOfMonths_PartBCov','Race','State']
    for i in range(1,11):
        remove_col.append('ClmDiagnosisCode_'+str(i))
    for i in range(1,7):
        remove_col.append('ClmProcedureCode_'+str(i))
        
    X_train=X_train.drop(columns=remove_col,axis=1)
    
    
    print(X_train.shape)
   
    
    print('normalization started')
    Cont_col=['InscClaimAmtReimbursed', 'DeductibleAmtPaid','Admitted_days', 'Claim_time',
              'Amount_get', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
              'OPAnnualDeductibleAmt',  'Age', 'Tolat_chronic_cond', 'Total_ip_op_amount_reimb', 'total_ip_op_amount_deduct', 
              'Total_physican_attended', 
              'Total_ClmDiagnosisCode', 'Total_ClmProcedureCode','Diff_max_IPAnnualReimbursementAmt',
              'Diff_max_OPAnnualReimbursementAmt', 'Diff_max_InscClaimAmtReimbursed']

    for i in Cont_col:
        data_train=np.array(X_train[i]).reshape(-1,1)
        
        X_train[i]=normalize(data_train,axis=0).ravel()
    import sklearn
    def predict_with_best_t(proba, threshould):
        predictions = []
        for i in proba:
            if i>=threshould:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
    print('model_loaded')  
    X_train['RenalDiseaseIndicator']=X_train['RenalDiseaseIndicator'].astype(str).astype(int)
    print(os.getcwd())
    print('path is over')


    # model=joblib.load('/Deployement on Heroku/Best_clf.pkl')

    # path_to_model = os.path.join('Deployement on Heroku', 'Best_clf.pkl')
    # model = joblib.load(path_to_model)
    model=joblib.load('Best_clf.pkl')

    y_predict_tr=model.predict_proba(X_train)[:,1]
    prediction_tr=predict_with_best_t(y_predict_tr,0.382)
    pred_df=pd.DataFrame(prediction_tr,columns=['Prediction'])
    print('prediction done')
    return  pred_df,providr_id


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/index.html')
# def login():
#     return render_template('index.html')

@app.route('/home.html', methods=['POST'])
def login():
    # Process the form data and perform any necessary validation
    # If the form is valid, redirect to the home page
    return render_template('home.html')

@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/register.html')
def register():
    return render_template('register.html')

# @app.route('/upload.html')
# def upload():
#     return render_template('upload.html')

# @app.route('/upload.html#home')
# def homee():
#     return render_template('home.html')

# @app.route('/upload.html#about')
# def about():
#     return render_template('about.html')

@app.route('/upload.html', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        try:
            file_bene = request.files.get('beneficiary')
            temp_bene = pd.read_csv(file_bene)
            # if temp_bene is not None:
            #     tb = 'Yeah'
            # else:
            #     tb = 'NO'
            file_inp = request.files.get('inpatient')
            temp_inp = pd.read_csv(file_inp)
            # if temp_inp is not None:
            #     ti = 'Yeah'
            # else:
            #     ti = 'NO'
            file_out = request.files.get('outpatient')
            temp_out = pd.read_csv(file_out)
            # if temp_out is not None:
            #     to = 'Yeah'
            # else:
            #     to = 'NO'

            df,id =final_pipeline(temp_bene,temp_inp,temp_out)
            # df.to_csv('df.csv')
            # id.to_csv('id.csv')
            count=0
            for i in range(len(df)):
                if(df['Prediction'][i]==1):
                    count+=1
            fraud_count='Number of Fraud Providers : '+str(count)
            ham_count= 'Number of Non-Fraud Providers : '+str(len(df)-count)
            percent = 'Percentage of the Fraud Providers is '+str((count/len(df))*100)
            # index=0
            # if int(df['Prediction'][index])==1:
            #     string='Provider Id : '+str(str(id[index]))+'  is Fraud'
        
            # else:
            #     string='Provider Id : '+str(str(id[index]))+'  is not Fraud'

            return render_template('upload.html',b=fraud_count,c=ham_count,d=percent)
        except Exception as e:
            return render_template('error.html', error=str(e))
    else:
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)


