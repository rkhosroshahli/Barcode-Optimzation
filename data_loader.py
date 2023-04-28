import numpy as np
import pandas as pd
import pickle


def load_data(excel_link: str, dict_labels: dict, fl_p1: str, fl_p2: str):
    # fl_p1 = 'DenseNet121_features'
    # fl_p2 = '_DN121_features_dict'
    # excel_link = "kimiaNet_train_data.xlsx"

    df = pd.read_excel(excel_link)

    X_data, y_data = [], []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if not row['project_id'] in dict_labels.keys():
            continue
        try:
            file_link = f'./{fl_p1}/' + row.slide_name + f'_{fl_p2}.pickle'
            pickle_file = pickle.load(open(file_link, 'rb'))
        except:
            continue

        fv = np.array(list(pickle_file.values()))
        if len(fv) == 0:
            continue
        mean_fv = np.mean(fv, axis=0)
        X_data.append(mean_fv)

        label = dict_labels[row.project_id]
        y_data.append(label)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data
