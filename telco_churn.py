import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load():
    data = pd.read_csv("datasets/Telco-Customer-Churn.csv")
    return data


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Types #####################")
    print(dataframe.info())
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### MissingValues #######################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    It provides the names of the categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Categorical variables also include numerically represented categorical variables.
    
    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be obtained.
        cat_th: int, optional
                For numerical but categorical variables, the class threshold value.
        car_th: int, optinal
                For categorical but cardinal variables, the class threshold value.

    Returns
    ------
        cat_cols: list
                List of categorical variables.
        num_cols: list
                List of numerical variables.
        cat_but_car: list
                List of cardinal variables that appear categorical.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = Total number of variables.
        num_but_cat in cat_cols
        The sum of the three lists returned (cat_cols + num_cols + cat_but_car) is equal to the total number of variables.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"CHURN_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def target_vs_category_visual(dataframe, target, categorical_col):
    plt.figure(figsize=(15, 8))
    sns.histplot(x=target, hue=categorical_col, data=dataframe, element="step", multiple="dodge")
    plt.title("Kategorik Değişkenlerin Churn'e Göre Durumu")
    plt.show()


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def model_olusturma(dataframe, model, target):
    y = dataframe[target]
    X = dataframe.drop(columns=target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
    rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(model, "için sonuç doğruluk değeri:", acc)
    return acc

######################################
df = load()
df.head()
check_df(df)

df.info()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df = df.drop(["customerID"], axis=1)

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


for col in cat_cols:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col)


df["Churn"].value_counts()
df['Churn'].value_counts()*100.0/len(df)

df['SeniorCitizen'].value_counts()*100.0/len(df)

df['Contract'].value_counts()
df['Contract'].value_counts()/len(df)

sns.boxplot(x=df.Churn, y=df.tenure)
plt.show()

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

for col in cat_cols:
    target_vs_category_visual(df, "Churn", col)


for col in num_cols:
    target_summary_with_num(df, "Churn", col)

for col in num_cols:
    df.groupby('Churn').agg({col: 'mean'}).plot(kind='bar', rot=0, figsize=(16, 8))


for col in num_cols:
    print(col, check_outlier(df, col))


for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


missing_values_table(df)
missing_values_table(df, True)

cor_matrix = df[num_cols].corr()
print(cor_matrix)

# Feature Engineering

na_cols = missing_values_table(df, True)
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
# df.dropna(inplace=True)

for col in cat_cols:
    cat_summary(df, col)


df["gender"] = df["gender"].map({'Male': 0, 'Female': 1})
df.groupby("gender").agg({"Churn": ["mean", "count"]})
df.loc[((df['gender'] == 0) & (df["SeniorCitizen"] == 1)), 'SENIOR/YOUNG_GENDER'] = "senior_male"
df.loc[((df['gender'] == 0) & (df["SeniorCitizen"] == 0)), 'SENIOR/YOUNG_GENDER'] = "young_male"
df.loc[((df['gender'] == 1) & (df["SeniorCitizen"] == 1)), 'SENIOR/YOUNG_GENDER'] = "senior_female"
df.loc[((df['gender'] == 1) & (df["SeniorCitizen"] == 0)), 'SENIOR/YOUNG_GENDER'] = "young_female"
df.groupby("SENIOR/YOUNG_GENDER").agg({"Churn": ["mean", "count"]})

df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"
df.groupby("NEW_TENURE_YEAR").agg({"Churn": ["mean", "count"]})


df.groupby("PhoneService").agg({"Churn": ["mean", "count"]})
df.loc[((df['gender'] == 0) & (df["PhoneService"] == "Yes")), 'PHONE_SER_GENDER'] = "phone_service_male"
df.loc[((df['gender'] == 0) & (df["PhoneService"] == "No")), 'PHONE_SER_GENDER'] = "no_phone_service_male"
df.loc[((df['gender'] == 1) & (df["PhoneService"] == "Yes")), 'PHONE_SER_GENDER'] = "phone_service_female"
df.loc[((df['gender'] == 1) & (df["PhoneService"] == "No")), 'PHONE_SER_GENDER'] = "no_phone_service_female"
df.groupby("PHONE_SER_GENDER").agg({"Churn": ["mean", "count"]})


df.groupby("Dependents").agg({"Churn": ["mean", "count"]})
df.loc[((df['gender'] == 0) & (df["Dependents"] == "Yes")), 'DEPEND_GENDER'] = "dependent_male"
df.loc[((df['gender'] == 0) & (df["Dependents"] == "No")), 'DEPEND_GENDER'] = "undependent_male"
df.loc[((df['gender'] == 1) & (df["Dependents"] == "Yes")), 'DEPEND_GENDER'] = "dependent_female"
df.loc[((df['gender'] == 1) & (df["Dependents"] == "No")), 'DEPEND_GENDER'] = "undependent_female"
df.groupby("DEPEND_GENDER").agg({"Churn": ["mean", "count"]})


df.groupby("MultipleLines").agg({"Churn": ["mean", "count"]})
df.loc[((df['gender'] == 0) & (df["MultipleLines"] == "Yes")), 'PHONE_LINE_GENDER'] = "multiple_lines_male"
df.loc[((df['gender'] == 0) & (df["MultipleLines"] == "No")), 'PHONE_LINE_GENDER'] = "single_line_male"
df.loc[((df['gender'] == 0) & (df["MultipleLines"] == "No phone service")), 'PHONE_LINE_GENDER'] = "no_line_male"
df.loc[((df['gender'] == 1) & (df["MultipleLines"] == "Yes")), 'PHONE_LINE_GENDER'] = "multiple_lines_female"
df.loc[((df['gender'] == 1) & (df["MultipleLines"] == "No")), 'PHONE_LINE_GENDER'] = "single_line_female"
df.loc[((df['gender'] == 1) & (df["MultipleLines"] == "No phone service")), 'PHONE_LINE_GENDER'] = "no_line_female"
df.groupby("PHONE_LINE_GENDER").agg({"Churn": ["mean", "count"]})


df.groupby("PaymentMethod").agg({"Churn": ["mean", "count"]})
df.loc[((df['gender'] == 0) & (df["PaymentMethod"] == "Electronic check")), 'GENDER_PAYMENT'] = "male_electronic_check_pay"
df.loc[((df['gender'] == 1) & (df["PaymentMethod"] == "Electronic check")), 'GENDER_PAYMENT'] = "female_electronic_check_pay"
df.groupby("GENDER_PAYMENT").agg({"Churn": ["mean", "count"]})


df.groupby("Contract").agg({"Churn": ["mean", "count"]})
df.loc[((df['gender'] == 0) & (df["Contract"] == "Month-to-month")), 'GENDER_CONTRACT'] = "male_monthly_contract"
df.loc[((df['gender'] == 1) & (df["Contract"] == "Month-to-month")), 'GENDER_CONTRACT'] = "female_monthly_contract"
df.groupby("GENDER_CONTRACT").agg({"Churn": ["mean", "count"]})


df.groupby("InternetService").agg({"Churn": ["mean", "count"]})
df.loc[((df['SENIOR/YOUNG_GENDER'] == "senior_male") & (df["InternetService"] == "Fiber optic")), 'SEN_YNG_GENDER_FIBER'] = "senior_male_fiber_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "young_male") & (df["InternetService"] == "Fiber optic")), 'SEN_YNG_GENDER_FIBER'] = "young_male_fiber_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "senior_female") & (df["InternetService"] == "Fiber optic")), 'SEN_YNG_GENDER_FIBER'] = "senior_female_fiber_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "young_female") & (df["InternetService"] == "Fiber optic")), 'SEN_YNG_GENDER_FIBER'] = "young_female_fiber_internet"
df.groupby("SEN_YNG_GENDER_FIBER").agg({"Churn": ["mean", "count"]})


df.groupby("OnlineSecurity").agg({"Churn": ["mean", "count"]})
df.loc[((df['gender'] == 0) & (df["InternetService"] == "Fiber optic") & (df["OnlineSecurity"] == "No")), 'INT_SEC_SERV_GENDER'] = "male_fiber_int_no_security"
df.loc[((df['gender'] == 1) & (df["InternetService"] == "Fiber optic") & (df["OnlineSecurity"] == "No")), 'INT_SEC_SERV_GENDER'] = "female_fiber_int_no_security"
df.groupby("INT_SEC_SERV_GENDER").agg({"Churn": ["mean", "count"]})


df.loc[((df['gender'] == 0) & (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")), 'INT_SEC_TECH_GENDER'] = "male_fiber_int_no_tech_sup"
df.loc[((df['gender'] == 1) & (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")), 'INT_SEC_TECH_GENDER'] = "female_fiber_int_no_tech_sup"
df.groupby("INT_SEC_TECH_GENDER").agg({"Churn": ["mean", "count"]})


df.loc[((df['gender'] == 0) & (df["Partner"] == "Yes")), 'PARTNER_GENDER'] = "with_partner_male"
df.loc[((df['gender'] == 0) & (df["Partner"] == "No")), 'PARTNER_GENDER'] = "without_partner_male"
df.loc[((df['gender'] == 1) & (df["Partner"] == "Yes")), 'PARTNER_GENDER'] = "with_partner_female"
df.loc[((df['gender'] == 1) & (df["Partner"] == "No")), 'PARTNER_GENDER'] = "without_partner_female"
df.groupby("PARTNER_GENDER").agg({"Churn": ["mean", "count"]})


df['new_totalservices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)


df["new_avg_charges"] = df["TotalCharges"] / (df["tenure"] + 1)


df["new_increase"] = df["new_avg_charges"] / df["MonthlyCharges"]


df["new_avg_service_fee"] = df["MonthlyCharges"] / (df['new_totalservices'] + 1)

# Encoding .
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Label Encoding
le = LabelEncoder()
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
print(binary_cols)

for col in binary_cols:
    df = label_encoder(df, col)
df.head()

# One-Hot Encoding
ohe_cols = [col for col in df.columns if 30 >= df[col].nunique() > 2]
dff = one_hot_encoder(df, ohe_cols)
dff.head()

# Scaling
scaler = StandardScaler()
dff[num_cols] = scaler.fit_transform(dff[num_cols])
dff.head()
for col in num_cols:
    dff[col] = RobustScaler().fit_transform(dff[[col]])

y = dff["Churn"]
X = dff.drop(["Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)


rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

model_olusturma(rf_model,accuracy_score)
