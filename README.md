## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="657" height="449" alt="371402519-48e55c78-b140-4c59-a597-4c2ffa10ba3c" src="https://github.com/user-attachments/assets/4104f1f0-b7cd-40fc-b3b4-40559356b091" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])```
```
<img width="267" height="191" alt="371402603-6202e425-9dac-4419-a7d9-b3d079c1281d" src="https://github.com/user-attachments/assets/f508f3b5-d56f-4190-9596-e670c9abb399" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="490" height="361" alt="371402775-30ad00dd-bd64-4e1c-a4aa-9e7ef6bba02a" src="https://github.com/user-attachments/assets/2e7e9535-3b5c-43e5-85c0-98519a5ab43b" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="1860" height="569" alt="371402858-28f0ee9e-203a-4f73-9eb5-110d7cc095b3" src="https://github.com/user-attachments/assets/5c31b869-1bef-4d51-9b4e-d9f87896b29e" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
<img width="526" height="351" alt="371403058-5d31c908-ff0f-4c56-851b-678e86af733f" src="https://github.com/user-attachments/assets/4a6ab8cb-c4b1-4ad9-bfbb-e72b4c092db0" />

```
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="1865" height="186" alt="371403057-1f6868a8-969b-4d48-886d-90bbb2f6ba85" src="https://github.com/user-attachments/assets/c67bc056-4209-4a60-ad85-bf20ee1abd83" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="762" height="343" alt="371403121-e5475c43-f6bf-484e-9efe-9f58fcb5656f" src="https://github.com/user-attachments/assets/f74c5e7f-aa37-4bff-b2d5-be251bef5867" />

```
pip install --upgrade category_encoders
```
<img width="1184" height="350" alt="371403195-786cd3e4-1ac6-4ddf-a9c8-e5b898a7ae0e" src="https://github.com/user-attachments/assets/5d49b79b-4ee9-4006-baf4-1cfa1ec31eca" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="682" height="413" alt="371403300-076d4e69-0db3-46c3-af19-bb79fde87021" src="https://github.com/user-attachments/assets/f21a6077-a13d-4b65-ab8d-b3713bafbdd1" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
<img width="760" height="419" alt="371403377-6b44ebc2-3f48-4132-b586-72bf5586569c" src="https://github.com/user-attachments/assets/4544566e-ef97-407c-adff-7c7e9e342427" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="754" height="422" alt="371403448-4b769be3-75a8-4c9e-b911-ef21f4b8f454" src="https://github.com/user-attachments/assets/7aa76647-b8d3-409f-a519-242d43f699e6" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
```
df.skew()
```

<img width="527" height="181" alt="371403692-c08080a9-b7f2-4386-98df-bb4e21925b16" src="https://github.com/user-attachments/assets/5a4ae39a-63c1-4103-8cf9-41b35236b72c" />

```
np.log(df["Highly Positive Skew"])
```
<img width="524" height="237" alt="371403740-9fcc0422-cf05-461c-b273-df5a8a47cc45" src="https://github.com/user-attachments/assets/a60b76c5-5e94-44a5-a62e-3aedb54151ff" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="564" height="231" alt="371403855-e9d6c413-5656-41f6-9d77-12e29c6005e3" src="https://github.com/user-attachments/assets/ed023735-1332-4dbb-86d1-1e5a648287ad" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="566" height="234" alt="371403940-bc8e70b9-eaf1-4328-a6cb-e7822062f7f5" src="https://github.com/user-attachments/assets/2d314047-3220-4aba-b328-f2641e3e9754" />

```
np.square(df["Highly Positive Skew"])
```

<img width="530" height="230" alt="371404155-1f640aca-1bd1-4a1a-8d53-42689b5d42d0" 
src="https://github.com/user-attachments/assets/ad5d2a67-3e13-4310-ae47-d41e9de944ea" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="897" height="448" alt="371404167-628eecf3-9067-42b3-adbf-d2598efbbac5" src="https://github.com/user-attachments/assets/2f0b7049-4535-40c8-a8ef-1903e6997d07" />

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```

<img width="891" height="467" alt="371404244-5bdfdcad-db7c-492c-94d1-7b38800e94f3" src="https://github.com/user-attachments/assets/f371d4b9-a86f-4326-8a02-4babe575dcbd" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="652" height="450" alt="371404330-8a437486-eee0-4c62-9ecd-33957b18541a" src="https://github.com/user-attachments/assets/558f11f2-a987-472d-93cf-167520073d52" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
<img width="688" height="464" alt="371404435-a6a6a3dd-25af-4778-93e6-15c599b0da53" src="https://github.com/user-attachments/assets/05f77f9b-e7b9-4378-9011-618fb5c607a6" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="891" height="459" alt="371404581-1ed95f51-a621-4d10-905d-91bbf2b5044d" src="https://github.com/user-attachments/assets/f00691eb-e5f7-4970-a028-0daca60e4a09" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="669" height="448" alt="371404698-477f7aad-f8a4-4e1e-98e5-f1c6ea6740c1" src="https://github.com/user-attachments/assets/ea5f9bc2-9c15-40b1-8118-9ae55dda7b84" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="621" height="449" alt="371405134-86c68bd7-fab7-4c68-99a3-01b826191813" src="https://github.com/user-attachments/assets/6dca46db-db2b-4cf7-9a26-d613184125f7" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="656" height="457" alt="371404851-93bd3f83-5292-43fb-a7e2-5ac084f4e157" src="https://github.com/user-attachments/assets/12b0e613-04ba-4e60-adcb-2abcd14609f5" />

# RESULT:

   Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
