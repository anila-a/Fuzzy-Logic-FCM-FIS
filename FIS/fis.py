import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define antecedents variables and consequent variable
x1 = ctrl.Antecedent(np.arange(1, 5, 1), 'Organizational Culture')
x2 = ctrl.Antecedent(np.arange(1, 5, 1), 'Involvement and Influence')
x3 = ctrl.Antecedent(np.arange(1, 5, 1), 'Communication and Practices')
x4 = ctrl.Antecedent(np.arange(0, 7, 1), 'Growth and Development')
x5 = ctrl.Antecedent(np.arange(0, 15, 1), 'Recognition and Reward')
x6 = ctrl.Antecedent(np.arange(1, 5, 1), 'Balance')
x7 = ctrl.Antecedent(np.arange(2500, 20000, 1), 'Revenue')

mh = ctrl.Consequent(np.arange(0, 101, 1), 'Need for Mental Health Treatment')

# Linguistic Variables
var_1 = ['Low', 'Medium', 'High', 'Very High']
var_2 = ['Bad', 'Good', 'Better', 'Best']

# Assign custom variables to membership functions
x1.automf(names=var_1)
x2.automf(names=var_1)
x3.automf(names=var_1)
x4.automf(names=var_1)
x5.automf(names=var_1)
x6.automf(names=var_2)
x7.automf(names=var_1)

# Triangular Membership Functions
x1['Low'] = fuzz.trimf(x1.universe, [1, 1, 2])
x1['Medium'] = fuzz.trimf(x1.universe, [1, 2, 3])
x1['High'] = fuzz.trimf(x1.universe, [2, 3, 4])
x1['Very High'] = fuzz.trimf(x1.universe, [3, 4, 5])

x2['Low'] = fuzz.trimf(x2.universe, [1, 1, 2])
x2['Medium'] = fuzz.trimf(x2.universe, [1, 2, 3])
x2['High'] = fuzz.trimf(x2.universe, [2, 3, 4])
x2['Very High'] = fuzz.trimf(x2.universe, [3, 4, 5])

x3['Low'] = fuzz.trimf(x3.universe, [1, 1, 2])
x3['Medium'] = fuzz.trimf(x3.universe, [1, 2, 3])
x3['High'] = fuzz.trimf(x3.universe, [2, 3, 4])
x3['Very High'] = fuzz.trimf(x3.universe, [3, 4, 5])

x4['Low'] = fuzz.trimf(x4.universe, [0, 0, 2])
x4['Medium'] = fuzz.trimf(x4.universe, [0, 2, 4])
x4['High'] = fuzz.trimf(x4.universe, [2, 4, 6])
x4['Very High'] = fuzz.trimf(x4.universe, [4, 6, 6])

x5['Very High'] = fuzz.trimf(x5.universe, [0, 0, 4.666])
x5['High'] = fuzz.trimf(x5.universe, [0, 4.666, 9.332])
x5['Medium'] = fuzz.trimf(x5.universe, [4.666, 9.332, 14])
x5['Low'] = fuzz.trimf(x5.universe, [9.332, 14, 14])

x6['Bad'] = fuzz.trimf(x6.universe, [1, 1, 2])
x6['Good'] = fuzz.trimf(x6.universe, [1, 2, 3])
x6['Better'] = fuzz.trimf(x6.universe, [2, 3, 4])
x6['Best'] = fuzz.trimf(x6.universe, [3, 4, 4])

x7['Low'] = fuzz.trimf(x7.universe, [2500, 2500, 8333])
x7['Medium'] = fuzz.trimf(x7.universe, [2500, 8333, 14166])
x7['High'] = fuzz.trimf(x7.universe, [8333, 14166, 20000])
x7['Very High'] = fuzz.trimf(x7.universe, [14166, 20000, 20000])

mh['Low'] = fuzz.trimf(mh.universe, [0, 0, 50])
mh['Moderate'] = fuzz.trimf(mh.universe, [10, 50, 90])
mh['High'] = fuzz.trimf(mh.universe, [50, 100, 100])

# Define a set of rules
rule1 = ctrl.Rule(x1['Very High'] & x2['Very High'] & x3['Very High'] & x4['Very High'] & x5['Very High']
                  & x6['Best'] & x7['Very High'], mh['Low'])
rule2 = ctrl.Rule(x1['Low'] & x2['Low'] & x3['Low'] & x4['Low'] & x5['Low'] & x6['Bad'] & x7['Low'], mh['High'])
rule3 = ctrl.Rule(x1['Medium'] & x2['Medium'] & x3['Medium'] & x4['Medium'] & x5['Medium'] & x6['Good'] & x7['Medium'], mh['Moderate'])

# Control system simulation
output_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
output = ctrl.ControlSystemSimulation(output_ctrl)

# Sample inputs
output.input['Organizational Culture'] = 4
output.input['Involvement and Influence'] = 4
output.input['Communication and Practices'] = 4
output.input['Growth and Development'] = 13
output.input['Recognition and Reward'] = 0
output.input['Balance'] = 4
output.input['Revenue'] = 20000

# Compute
output.compute()
print(output.output['Need for Mental Health Treatment'])
