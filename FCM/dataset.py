'''
 0   Age
 1   EnvironmentSatisfaction   [1-4] (Low, Medium, High, Very High)
 2   JobInvolvement   [1-4] (Low, Medium, High, Very High)
 3   MonthlyIncome   [2500-20000] (Low, Medium, High, Very High)
 4   RelationshipSatisfaction   [1-4] (Low, Medium, High, Very High)
 5   TrainingTimesLastYear   [0-6] (Low, Medium, High, Very High)
 6   WorkLifeBalance   [1 - 4] (Bad, Good, Better, Best)
 7   YearsSinceLastPromotion   [0-14] (Low, Medium, High, Very High)
'''

import pandas as pd
import numpy as np

# Read the dataset
data = pd.read_csv('dataset.csv')

# Drop unuseful columns
data.drop(columns = ['Attrition', 'BusinessTravel', 'DailyRate', 'Department',
                    'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
                    'EmployeeNumber', 'Gender', 'HourlyRate', 'JobLevel', 'JobRole',
                    'JobSatisfaction', 'MaritalStatus', 'MonthlyRate', 'NumCompaniesWorked',
                    'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
                    'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'YearsAtCompany',
                    'YearsInCurrentRole', 'YearsWithCurrManager'], inplace = True)

# Columns' information stored in arrays D1-D8
mh1 = pd.DataFrame(data, columns = ['Age'])
mh2 = pd.DataFrame(data, columns = ['EnvironmentSatisfaction'])
mh3 = pd.DataFrame(data, columns = ['JobInvolvement'])
mh4 = pd.DataFrame(data, columns = ['MonthlyIncome'])
mh5 = pd.DataFrame(data, columns = ['RelationshipSatisfaction'])
mh6 = pd.DataFrame(data, columns = ['TrainingTimesLastYear'])
mh7 = pd.DataFrame(data, columns = ['WorkLifeBalance'])
mh8 = pd.DataFrame(data, columns = ['YearsSinceLastPromotion'])

# Combine columns to create matrices of size 2x2
# Use arrays for clustering
mh_x1 = np.append(mh2, mh1, axis=1)
mh_x2 = np.append(mh3, mh1, axis=1)
mh_x3 = np.append(mh4, mh1, axis=1)
mh_x4 = np.append(mh5, mh1, axis=1)
mh_x5 = np.append(mh6, mh1, axis=1)
mh_x6 = np.append(mh7, mh1, axis=1)
mh_x7 = np.append(mh8, mh1, axis=1)
