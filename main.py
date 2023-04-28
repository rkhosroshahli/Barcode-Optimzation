from barcode_optimizer import barcode_optimizer
from data_loader import load_data

dict_labels = {'TCGA-GBM': 0, 'TCGA-LGG': 1}  # Brain
# dict_labels={'TCGA-LUAD':0, 'TCGA-LUSC':1} #  Lung
sel_features = [887, 874, 900]


"""X_train, y_train = load_data(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels,
                             fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
X_test, y_test = load_data(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels,
                           fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
X_val, y_val = load_data(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels,
                         fl_p1="AllDensePatches", fl_p2="DN121_features_dict")"""

X_train, y_train = load_data(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels,
                             fl_p1="KimiaNet_Features", fl_p2="KimiaNet_features_dict")
X_test, y_test = load_data(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels,
                           fl_p1="KimiaNet_Features", fl_p2="KimiaNet_features_dict")
X_val, y_val = load_data(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels,
                         fl_p1="KimiaNet_Features", fl_p2="KimiaNet_features_dict")


#save_file_link=f"Brain_SelectedFeatures_DenseNet121_MaxGens{num_generations}_NP{num_population}_K{K}_Runs{num_runs}.npz"
save_file_link_p1=f"Brain_SelectedFeatures_DenseNet121"
save_file_link_p1=f"Brain_SelectedFeatures_KimiaNet"
barcode_optimizer(dict_labels, sel_features, save_file_link_p1, X_train, y_train, X_test, y_test, X_val, y_val)
