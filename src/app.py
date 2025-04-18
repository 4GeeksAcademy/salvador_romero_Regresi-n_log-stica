
def eliminar_atipicos(datos, columnas):
    new_data = datos
    for i in columnas:
        q1=new_data[i].quantile(0.25)
        q3=new_data[i].quantile(0.75)
        iqr = q3-q1
        low_lim = q1 - 1.5*iqr
        hi_lim = q3 + 1.5*iqr
        rem = new_data[(new_data[i]>=hi_lim) | (new_data[i]< low_lim)]
        new_data = new_data.drop(index=rem.index)
    return new_data.copy()

# your code here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep=';')
data.info()
data.to_csv("./data/raw/data.csv",index=False)

data_uniques = data.drop_duplicates()

col = ["age","duration","campaign","emp.var.rate"]


data_filter = eliminar_atipicos(data_uniques,col)

data_filter['job']=pd.factorize(data_filter["job"])[0]
data_filter['marital']=pd.factorize(data_filter["marital"])[0]
data_filter['education']=pd.factorize(data_filter["education"])[0]
data_filter['default']=pd.factorize(data_filter["default"])[0]
data_filter['housing']=pd.factorize(data_filter["housing"])[0]
data_filter['loan']=pd.factorize(data_filter["loan"])[0]
data_filter['contact']=pd.factorize(data_filter["contact"])[0]
data_filter['month']=pd.factorize(data_filter["month"])[0]
data_filter['day_of_week']=pd.factorize(data_filter["day_of_week"])[0]
data_filter['poutcome']=pd.factorize(data_filter["poutcome"])[0]
data_filter['y']=pd.factorize(data_filter["y"])[0]

col = ["pdays", "emp.var.rate","euribor3m", "nr.employed", "y"]
data_filter = data_filter[col]
X = data_filter.drop("y",axis=1)
Y = data_filter["y"]
x_tr, x_tst, y_tr, y_tst = train_test_split(X, Y, test_size=0.2, random_state=8)



# sel_model = SelectKBest(chi2,k=4)
# sel_model.fit(x_tr,y_tr)
# ix = sel_model.get_support()
# x_train_sel = pd.DataFrame(sel_model.transform(x_tr),columns= x_tr.columns.values[ix])
# x_test_sel = pd.DataFrame(sel_model.transform(x_tst),columns= x_tst.columns.values[ix])

# print(x_train_sel.head())


model = LogisticRegression(max_iter=1000)
model.fit(x_tr,y_tr)

y_pred = model.predict(x_tst)

grid_accuracy = accuracy_score(y_tst, y_pred)
print(grid_accuracy)