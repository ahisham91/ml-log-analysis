import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

os.getcwd()
PATH = r'C:\Users\anish\Downloads'
fname = 'usa_00001.csv'
fname1 = 'MP01-Crosswalk.csv'
df = pd.read_csv(os.path.join(PATH, fname))
columns_to_drop = [1, 3, 5]
df.drop(df.columns[columns_to_drop], axis=1, inplace=True)
df.head()
crosswalk = pd.read_csv(os.path.join(PATH, fname1))
crosswalk.head()
crosswalk.rename(columns={'educd': 'EDUCD', 'educdc': 'EDUCDC'}, inplace=True)
df_final = pd.merge(df, crosswalk[['EDUCD', 'EDUCDC']], on='EDUCD', how='left')
df_final.shape
df_final['hsdip'] = df_final['EDUCD'].apply(lambda x: 1 if x in [62, 63, 64, 65, 70, 71, 80, 81, 82, 83, 90, 100] else 0)
df_final['coldip'] = df_final['EDUCD'].apply(lambda x: 1 if x in [101, 114, 115, 116] else 0)
df_final['white'] = df_final['RACED'].apply(lambda x: 1 if x in [100] else 0)
df_final['black'] = df_final['RACED'].apply(lambda x: 1 if x in [200] else 0)
df_final['hispanic'] = df_final['HISPAN'].apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)
df_final['married'] = df_final['MARST'].apply(lambda x: 1 if x in [1, 2] else 0)
df_final['female'] = df_final['SEX'].apply(lambda x: 1 if x in [2] else 0)
df_final['VET'] = df_final['VETSTAT'].apply(lambda x: 1 if x in [2] else 0)
df_final['hsdip_EDUCDC'] = df_final['hsdip'] * df_final['EDUCDC']
df_final['coldip_EDUCDC'] = df_final['coldip'] * df_final['EDUCDC']
df_final['AGE_squared'] = df_final['AGE'] ** 2
df_final = df_final[df_final['INCWAGE'] > 0]
df_final['LN_INCWAGE'] = np.log(df_final['INCWAGE'])

#CHECKS
 
has_999 = df_final['EDUCD'].isin([999]).any()
print(f"Is 999 present in EDUCD? {has_999}")
has_900 = df_final['HISPAND'].isin([900]).any()
print(f"Is 900 present in HISPAND? {has_900}")

###############################################################
 
#Q1

pd.set_option('display.max_columns', None)
df_final.columns
columns_of_interest = ['YEAR', 'INCWAGE', 'LN_INCWAGE', 'EDUCDC', 'female', 'AGE', 
                       'AGE_squared', 'white', 'black', 'hispanic', 'married', 
                       'NCHILD', 'VET', 'hsdip', 'coldip', 'hsdip_EDUCDC', 'coldip_EDUCDC']
df_final[columns_of_interest].describe()
 
#Q2

plt.figure(figsize=(10, 6))
sns.scatterplot(x='EDUCDC', y='LN_INCWAGE', data=df_final)
sns.regplot(x='EDUCDC', y='LN_INCWAGE', data=df_final, scatter=False, color='red')
plt.xlabel('Education Years')
plt.ylabel('Logarithm of Income Wage')
plt.title('Logarithm of Income Wage vs Education Years')
plt.show()
 
#Q3

X = df_final[['EDUCDC', 'female', 'AGE', 'AGE_squared', 'white', 'black', 
              'hispanic', 'married', 'NCHILD', 'VET']]
y = df_final['LN_INCWAGE']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
model_summary = model.summary()
model_summary

# Age for maximum wage 
beta_age = model.params['AGE']
beta_age_squared = model.params['AGE_squared']

# Age for maximum wage = -?3 / (2 * ?4)
age_max_wage = -beta_age / (2 * beta_age_squared)
age_max_wage
 
#Q4

df_final['Education_Category'] = 'No High School Diploma'
df_final.loc[(df_final['hsdip_EDUCDC'] > 0) & (df_final['coldip_EDUCDC'] ==
0), 'Education_Category'] = 'High School Diploma'
df_final.loc[df_final['coldip_EDUCDC'] > 0, 'Education_Category'] = 'College Degree'
palette = {'No High School Diploma': 'green', 'High School Diploma': 'blue', 'College Degree': 'orange'}
plt.figure(figsize=(12, 8))
sns.scatterplot(x='EDUCDC', y='LN_INCWAGE', data=df_final, hue='Education_Category', palette=palette)
sns.lmplot(x='EDUCDC', y='LN_INCWAGE', data=df_final, hue='Education_Category', palette=palette, height=8, aspect=1.5)
plt.xlabel('Education Years')
plt.ylabel('Logarithm of Income Wage')
plt.title('Logarithm of Income Wage vs Education Years for Different Education Categories')
plt.show()
 
#Q6

#df_final['hsdip'] = (df_final['hsdip_EDUCDC'] > 0).astype(int)
#df_final['coldip'] = (df_final['coldip_EDUCDC'] > 0).astype(int)
X_new_model = df_final[['EDUCDC', 'female', 'AGE', 'AGE_squared', 'white', 'black', 'hispanic', 
                      'married', 'NCHILD', 'VET', 'hsdip', 'coldip', 'hsdip_EDUCDC', 'coldip_EDUCDC']]
y = df_final['LN_INCWAGE']
X_new_model = sm.add_constant(X_new_model)
new_model = sm.OLS(y, X_new_model).fit()
new_model_summary = new_model.summary()
new_model_summary

 #Q7
df_final['AGE_cubed'] = df_final['AGE'] **3
df_final['EDUCDC_squared'] = df_final['EDUCDC'] ** 2
df_final['EDUCDC_female'] = df_final['EDUCDC'] * df_final['female']
df_final['EDUCDC_black'] = df_final['EDUCDC'] * df_final['black']
df_final['EDUCDC_hispanic'] = df_final['EDUCDC'] * df_final['hispanic']
df_final['EDUCDC_married'] = df_final['EDUCDC'] * df_final['married']
X_model3 = df_final[['EDUCDC', 'EDUCDC_squared', 'female', 'AGE',
 'AGE_squared', 'AGE_cubed', 'white', 'black', 'hispanic', 'married',
 'NCHILD', 'VET', 'hsdip', 'coldip', 'hsdip_EDUCDC', 'coldip_EDUCDC',
 'EDUCDC_female', 'EDUCDC_black', 'EDUCDC_hispanic', 'EDUCDC_married']]
X_model3 = sm.add_constant(X_model3)
model3 = sm.OLS(y, X_model3).fit()
model3_adj_r2 = model3.rsquared_adj
best_model_summary = model3.summary()
best_model_summary
