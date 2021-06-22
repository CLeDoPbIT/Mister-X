import os

path_data = '/shared/users/ashilov2/colleguches/data'
path_output  = '/home/angelaburova/covid_data'
path_row_train_data = os.path.join(path_data, 'train.csv')
path_row_test_data = os.path.join(path_data, 'test.csv')

# path_train_filtered_data = os.path.join(path_data, 'train.csv')
# path_test_filtered_data = os.path.join(path_data, 'test.csv')

path_train_filtered_data = os.path.join(path_data, 'train_cut_v3.csv')


path_train_interpolate_data_NO_cutted_pairs = os.path.join(path_data, 'train_interpolate_withoit_cut_pairs.csv')
path_test_interpolate_data_NO_cutted_pairs = os.path.join(path_data, 'test_interpolate_withoit_cut_pairs.csv')

path_train_interpolate = os.path.join('/shared/users/ashilov2/colleguches/data/train_interpolate_new.csv')
path_test_interpolate = os.path.join('/shared/users/ashilov2/colleguches/data/test_interpolate_new.csv')

path_augmented_data = os.path.join(path_data, 'train_cut_v3_generated.npz')
augmented_train_interpolate = os.path.join(path_output, 'augmented_interpolate_data.csv')