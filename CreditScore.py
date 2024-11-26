import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
pio.templates.default = "plotly_dark"

data = pd.read_csv('train.csv')
# print(data.head())

"""
Important Data:
Occupation
Annual_Income
Monthly_Inhand_Salary
Num_Bank_Accounts
Num_Credit_Card
Interest_Rate
Num_of_Loan
Type_of_Loan
Delay_from_due_date
Num_of_Delayed_Payment
Changed_Credit_Limit
Num_Credit_Inquiries
Credit_Mix
Outstanding_Debt
Credit_Utilization_Ratio
Credit_History_Age
Payment_of_Min_Amount
Total_EMI_per_Month
Amount_invested_monthly
Payment_Behaviour
Monthly_Balance
"""

""" fig = px.box(data, 
            x = "Occupation",
            color = "Credit_Score",
            title = "Credit Score Based on Occupation",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Shows that occupation does not affect credit score - won't use in model

""" fig = px.box(data, 
            x = "Credit_Score",
            y = "Annual_Income",
            color = "Credit_Score",
            title = "Credit Score Based on Annual Income",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Shows that annual income does affect credit score - will use in model, higher income = higher credit score

""" fig = px.box(data, 
            x = "Credit_Score",
            y = "Monthly_Inhand_Salary",
            color = "Credit_Score",
            title = "Credit Score Based on Monthly Inhand Salary",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Shows that monthly inhand salary does affect credit score - will use in model, higher salary = higher credit score

""" fig = px.box(data, 
            x = "Credit_Score",
            y = "Num_Bank_Accounts",
            color = "Credit_Score",
            title = "Credit Score Based on Number of Bank Accounts",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Shows that number of bank accounts does affect credit score - will use in model, more accounts = lower credit score

""" fig = px.box(data, 
            x = "Credit_Score",
            y = "Num_Credit_Card",
            color = "Credit_Score",
            title = "Credit Score Based on Number of Credit Cards",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Shows that number of credit cards does affect credit score - will use in model, more cards = lower credit score

""" fig = px.box(data, 
            x = "Credit_Score",
            y = "Interest_Rate",
            color = "Credit_Score",
            title = "Credit Score Based on Interest Rate",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Shows that interest rate does affect credit score - will use in model, higher rate = lower credit score

""" fig = px.box(data, 
            x = "Credit_Score",
            y = "Num_of_Loan",
            color = "Credit_Score",
            title = "Credit Score Based on Number of Loans",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Shows that number of loans does affect credit score - will use in model, more loans = lower credit score

""" fig = px.box(data, 
            x = "Type_of_Loan",
            color = "Credit_Score",
            title = "Credit Score Based on Occupation",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Shows that type of loan does affect credit score - will use in model, different loans = different credit scores

""" fig = px.box(data, 
            x = "Credit_Score",
            y = "Delay_from_due_date",
            color = "Credit_Score",
            title = "Credit Score Based on Delay from Due Date",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Shows that delay from due date does affect credit score - will use in model, more delay = lower credit score

""" fig = px.box(data, 
            x = "Credit_Score",
            y = "Num_of_Delayed_Payment",
            color = "Credit_Score",
            title = "Credit Score Based on Number of Delayed Payments",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Number of delayed payments affects credit score, Higher number = lower credit score

""" fig = px.box(data,
            x = "Credit_Score",
            y = "Changed_Credit_Limit",
            color = "Credit_Score",
            title = "Credit Score Based on Changed Credit Limit",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
             )
pyo.plot(fig) """
# No clear correlation between changed credit limit and credit score

""" fig = px.box(data,
            x = "Credit_Score",
            y = "Num_Credit_Inquiries",
            color = "Credit_Score",
            title = "Credit Score Based on Number of Credit Inquiries",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Number of credit inquiries affects credit score, Higher number = lower credit score

""" fig = px.box(data,
            x = "Credit_Score",
            y = "Outstanding_Debt",
            color = "Credit_Score",
            title = "Credit Score Based on Credit Mix",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# Outstanding debt affects credit score, Higher debt = lower credit score

""" fig = px.box(data,
            x = "Credit_Score",
            y = "Credit_Utilization_Ratio",
            color = "Credit_Score",
            title = "Credit Score Based on Credit Utilization Ratio",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            }
            )
pyo.plot(fig) """
# No clear correlation between credit utilization ratio and credit score

""" fig = px.box(data,
            x = "Credit_Score",
            y = "Credit_History_Age",
            color = "Credit_Score",
            title = "Credit Score Based on Credit History Age",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            } 
            )
pyo.plot(fig) """
# Credit history age affects credit score, Older credit history age = higher credit score

""" fig = px.box(data,
            x = "Credit_Score",
            y = "Total_EMI_per_month",
            color = "Credit_Score",
            title = "Credit Score Based on Total EMI per Month",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            } 
            )
pyo.plot(fig) """

""" fig = px.box(data,
            x = "Credit_Score",
            y = "Amount_invested_monthly",
            color = "Credit_Score",
            title = "Credit Score Based on Amount Invested Monthly",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            } 
            )
pyo.plot(fig) """
# Unclear correlation between amount invested monthly and credit score

""" fig = px.box(data,
            x = "Credit_Score",
            y = "Monthly_Balance",
            color = "Credit_Score",
            title = "Credit Score Based on Monthly Balance",
            color_discrete_map={
                'Poor': 'red',
                'Standard': 'yellow',
                'Good': 'green'
            } 
            )
pyo.plot(fig) """
# Monthly balance affects credit score, Higher balance = higher credit score

# Model will use: Annual_Income, Monthly_Inhand_Salary, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, 
# Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Credit Mix, Num_Credit_Inquiries, Outstanding_Debt, 
# Credit_History_Age, Monthly_Balance

# Creating the model

data["Credit_Mix"] = data["Credit_Mix"].map({"Standard": 1, "Poor": 0, "Good": 2})

x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", 
                "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", 
                "Num_of_Delayed_Payment", "Credit_Mix", "Num_Credit_Inquiries", 
                "Outstanding_Debt", "Credit_History_Age", "Monthly_Balance"]])
y = np.array(np.ravel(data[["Credit_Score"]]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Getting inputs from user to classify their credit score based on new data

print("Credit Score Classification: ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit Cards: "))
e = float(input("Interest Rate: "))
f = float(input("Number of Loans: "))
g = float(input("Delay from Due Date: "))
h = float(input("Number of Delayed Payments: "))
i = float(input("Credit Mix: "))
j = float(input("Number of Credit Inquiries: "))
k = float(input("Outstanding Debt: "))
l = float(input("Credit History Age: "))
m = float(input("Monthly Balance: "))

new_data = np.array([[a, b, c, d, e, f, g, h, i, j, k, l, m]])
print("Predicted Credit Score: ", model.predict(new_data))