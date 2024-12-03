import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ctypes
from sklearn.metrics import r2_score

def FitLinReg(x, y):
    return np.polyfit(x, y, 1)

df = pd.read_csv("game_skater_stats.csv", nrows=10000) # My computer is not particularly beefy

columnsWeCareAbout = [ "timeOnIce", "shots", "goals", "powerPlayTimeOnIce" ]
for column in df:
    if column in columnsWeCareAbout:
        for i, val in df[column].items():
            if val == 0:
                # Drop the row
                df.drop(index = i, inplace = True)

PPTimeOnIce = df["powerPlayTimeOnIce"].to_numpy()
timeOnIce = df["timeOnIce"].to_numpy()
shots = df["shots"].to_numpy()
goals = df["goals"].to_numpy()

shotsPerSecond = shots / timeOnIce
goalsPerSecond = goals / timeOnIce
shotsPerPPSecond = shots / PPTimeOnIce
goalsPerPPSecond = goals / PPTimeOnIce

m, b = np.polyfit(shotsPerSecond, goalsPerSecond, 1)
mModel2, bModel2 = np.polyfit(shotsPerPPSecond, goalsPerPPSecond, 1)

print(str(m) + " <- m, b -> " + str(b))
print(str(mModel2) + "<- m, b -> " + str(bModel2) + " (model 2)")

plt.title("SOG AFT vs Goals AFT by Players")
plt.xlabel("Shots Per Second")
plt.ylabel("Goals Per Second")
plt.scatter(shotsPerSecond, goalsPerSecond, marker = ".", label = "Actual Values")
# plt.plot(shotsPerSecond, goalsPerSecond, 'b.', shotsPerSecond, m * shotsPerSecond + b, "-k")
plt.legend(loc = "upper left")
plt.savefig("shotsPerSecond_x_goalsPerSecond.png")

plt.clf()

predictedYsModel2 = []

for shotCount in shotsPerPPSecond:
    predictedYsModel2.append((mModel2 * shotCount) + bModel2)

plt.title("PP SOG AFT vs PP Goals AFT by Players")
plt.xlabel("Shots Per Second On The Power Play")
plt.ylabel("Goals Per Second On The Power Play")
plt.scatter(shotsPerPPSecond, goalsPerPPSecond, marker = ".", label = "Actual Values")
# plt.plot(shotsPerPPSecond, goalsPerPPSecond, 'b.', shotsPerPPSecond, m * shotsPerPPSecond + b, "-k")
plt.legend(loc = "upper left")
plt.savefig("shotsPerPowerPlaySecond_x_goalsPerPowerPlaySecond.png")

plt.clf()

plt.title("PP SOG AFT vs PP Goals AFT by Players with Predicted Values")
plt.xlabel("Shots Per Second On The Power Play")
plt.ylabel("Goals Per Second On The Power Play")
plt.scatter(shotsPerPPSecond, goalsPerPPSecond, marker = ".", label = "Actual Values")
plt.scatter(shotsPerPPSecond, np.array(predictedYsModel2), c = "r", marker = ".", label = "Predicted Values")
# plt.plot(shotsPerPPSecond, goalsPerPPSecond, 'b.', shotsPerPPSecond, m * shotsPerPPSecond + b, "-k")
plt.legend(loc = "upper left")
plt.savefig("shotsPerPowerPlaySecond_x_goalsPerPowerPlaySecond_with_predictedValues.png")

plt.clf()

predictedYs = []

for shotCount in shotsPerSecond:
    predictedYs.append((m * shotCount) + b)
    
plt.title("SOG AFT vs Goals AFT by Players with Predicted Values")
plt.xlabel("Shots Per Second")
plt.ylabel("Goals Per Second")
plt.scatter(shotsPerSecond, goalsPerSecond, marker = ".", label = "Actual Values")
# plt.plot(shotsPerSecond, goalsPerSecond, 'b.', shotsPerSecond, m * shotsPerSecond + b, "-k")
plt.scatter(shotsPerSecond, np.array(predictedYs), c = "r", marker = ".", label = "Predicted Values")
plt.legend(loc = "upper left")
plt.savefig("shotsPerSecond_x_goalsPerSecond_with_predictedValues.png")

print("R^2 score: " + str(r2_score(goalsPerSecond, predictedYs)))
print("R^2 score of second model: " + str(r2_score(goalsPerPPSecond, predictedYsModel2)))