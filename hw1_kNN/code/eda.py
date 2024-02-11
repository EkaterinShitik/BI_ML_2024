import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt 


def run_eda(df: pd.DataFrame) -> None:
    """
    The function makes EDA of a dataframe of interest
    
    EDA includes:
    - the number of rows and the number of columns
    - types of data
    - main features of categorial variables
    - main features of numerical variables
    - the number of outliers in columns where they are relevant
    - analysis of missing values
    - a number of duplicate rows

    Arguments:
    - df (pd.DataFrame): a DataFrame of interest

    Return:
    - None
    """
    print("Hi there! Let's analyze your DataFrame 💫")
    print('')
    number_rows, number_columns = df.shape
    print(f"Number of rows: {number_rows}")
    print(f"Number of columns: {number_columns}")
    print('')
    types_of_data = {'categorical': [], 'numerical': [], 'string': [], 'object': []}
    for column in df.columns:
        if df[column].nunique() <= 5:
            df[column] = df[column].astype('category')
            types_of_data['categorical'].append(column)
        elif df[column].dtype == float or df[column].dtype == int:
           types_of_data['numerical'].append(column)
        elif df[column].dtype == object:
            if df[column].str.isnumeric().all():
                df[column] = df[column].astype('float') 
                types_of_data['numerical'].append(column)
            else:
                for element in df[column]:
                    if type(element) is not str and element is not np.nan: 
                        types_of_data['object'].append(column)
                        break
                else:
                   types_of_data['string'].append(column)
    print('The DataFrame contains the following number of variables:')
    common_inf = pd.DataFrame({'Categorical': [len(types_of_data['categorical'])], 
                               'Numerical': [len(types_of_data['numerical'])],
                               'String': [len(types_of_data['string'])],
                               'Object': [len(types_of_data['object'])]})
    display(common_inf)
    print('')
    print(f"Categorical: {types_of_data['categorical']}")
    print(f"Numerical: {types_of_data['numerical']}")
    print(f"String: {types_of_data['string']}")
    print(f"Object: {types_of_data['object']}")
    print('')
    print('Categorical data ✨')
    print('')
    for name in types_of_data['categorical']:
        print(f"{name} has {df[name].cat.categories.size} categories:")
        categoric_count = pd.DataFrame({'': [0, 0]}, index=['Count', 'Share'])
        for category in df[name].cat.categories:
            count_of_category = df[name][df[name] == category].count()
            freq_of_category = round(count_of_category / df[name].count(), 3)
            categoric_count[category] = [count_of_category, freq_of_category]
        categoric_count = categoric_count.drop(columns='')
        display(categoric_count)
    print('')
    print('Numerical data ✨')
    print('')
    for name in types_of_data['numerical']:
        min = df[name].min()
        max = df[name].max()
        mean = round(df[name].mean(), 3)
        std = round(df[name].std(), 3)
        median = round(df[name].median(), 3)
        q25 = df[name].quantile(0.25) 
        q75 = df[name].quantile(0.75)
        iqr = q75 - q25
        outliers_above = df[name][df[name] > (q75 + 1.5 * iqr)].size
        outliers_below = df[name][df[name] < (q25 - 1.5 * iqr)].size
        outliers = outliers_above + outliers_below
        print(f"{name} has the following statistics:")
        num_var_inf = pd.DataFrame({'Minimum': [min], 
                                    'Maximum': [max],
                                    'Mean': [mean],
                                    'Standard deviation': [std],
                                    'Mediana': [median],
                                    'Quantile 0.25': [q25],
                                    'Quantile 0.75': [q75],
                                    'Outliers': [outliers]})
        display(num_var_inf)
        figure, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.15, .85)})
        sns.boxplot(df[name], orient="h", ax=ax_box) 
        sns.histplot(df[name], ax=ax_hist)
        plt.show()
    print('Missing values in the DataFrame ✨')
    print('')
    print(f"Overall the DataFrame has {df.isna().sum().sum()} NA")
    count_na_string = 0
    for _ in range(df.shape[0]):
        if df.iloc[_].isna().sum() > 0:
            count_na_string += 1
    print(f"Missing values are in {count_na_string} strings")
    columns_with_na = []
    for _ in range(df.shape[1]):
        if df.isna().sum().iloc[_] > 0:
            columns_with_na += [df.columns[_]]
    print(f"Missing values are in columnes: {columns_with_na}")
