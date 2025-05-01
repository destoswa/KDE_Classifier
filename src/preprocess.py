import os
import pandas as pd
import open3d as o3d
import random


def preprocess(source_data, frac_train=.8, do_augment=False):
    remove_with_suffixe(source_data + '/Garbage', '_rot')
    remove_with_suffixe(source_data + '/Multi', '_rot')
    remove_with_suffixe(source_data + '/Single', '_rot')

    # Create file with different labels:
    with open(f'{source_data}/modeltrees_shape_names.txt', 'w') as f:
        f.write('garbage\nmultiple\nsingle')

    # Create references files:
    num_training_samples = 0
    num_testing_samples = 0
    df_training_samples = pd.DataFrame(columns=['data', 'label'])
    df_testing_samples = pd.DataFrame(columns=['data', 'label'])
    dic_temp = {}
    for folder in os.listdir(source_data):
        if os.path.isdir(source_data + "\\" + folder):
            label = 0
            if folder == 'Garbage':
                label = 0
            elif folder == 'Multi':
                label = 1
            elif folder == 'Single':
                label = 2

            # partition lists
            num_files = len(os.listdir(source_data + "/" + folder))
            num_train = int(num_files * frac_train)
            num_training_samples += num_train
            num_testing_samples += num_files - num_train
            data = os.listdir(source_data + "/" + folder)
            random.shuffle(data)

            # training data
            list_train = data[0:num_train]
            list_train = [folder + '/' + x for x in list_train]
            list_train_label = [label] * len(list_train)

            # testing data
            list_test = data[num_train::]
            list_test = [folder + '/' + x for x in list_test]
            list_test_label = [label] * len(list_test)

            # data to pandas
            dic_temp['data'] = list_train
            dic_temp['label'] = list_train_label
            df_temp = pd.DataFrame(dic_temp, columns=['data', 'label'])
            df_training_samples = pd.concat([df_training_samples, df_temp], ignore_index=True)
            dic_temp['data'] = list_test
            dic_temp['label'] = list_test_label
            df_temp = pd.DataFrame(dic_temp, columns=['data', 'label'])
            df_testing_samples = pd.concat([df_testing_samples, df_temp], ignore_index=True)

    if do_augment:
        print("Beginning data augmentation..")
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)   # Remove Open3D warnings
        repeat = 3
        df_training_samples = data_augmentation('Multi', df_training_samples, 90, repeat)
        df_training_samples = data_augmentation('Single', df_training_samples, 90, repeat)
        df_training_samples = data_augmentation('Garbage', df_training_samples, 90, repeat)

    print(f"Final partition for training : {num_training_samples}")
    print(f"Final partition for testing : {num_testing_samples}")
    df_training_samples.to_csv(f'{source_data}/modeltrees_train.csv', ';', index=False)
    df_testing_samples.to_csv(f'{source_data}/modeltrees_test.csv', ';', index=False)


def data_augmentation(src, df_training_samples, angle, repeat):
    print('data augmentation in folder : ' + src)
    list_samples = df_training_samples['data'].to_list()
    for file in os.listdir(src):
        full_file_name = src + '/' + file
        if full_file_name in list_samples:    # only augment training samples
            if not file.split('.')[0].endswith('_rot'): # test if the file is already a rotated file with this orientation
                pcd = o3d.t.io.read_point_cloud(src + '/' + file, format='pcd')
                for i in range(repeat):
                    angle_deg = angle * (i+1)
                    newfile = src+'/'+file.split('.')[0]+'_' + str(angle_deg) + '_rot.pcd'
                    pcd_new = pcd.clone()
                    new_pos = pcd.extrude_rotation(angle_deg, [0, 0, 1], resolution=1)
                    new_pos = new_pos.point['positions'][pcd_new.point['positions'].shape[0]:, :]
                    pcd_new.point['positions'] = new_pos
                    o3d.t.io.write_point_cloud(newfile, pcd_new, write_ascii=True)

                    # add new file to dataframe:
                    new_row = {'data': newfile, 'label': df_training_samples[df_training_samples['data'] == full_file_name]['label'].values[0]}
                    df_training_samples.loc[len(df_training_samples)] = new_row

    # update csv file
    print('data augmentation terminated')
    return df_training_samples


def remove_with_suffixe(src, suff):
    for file in os.listdir(src):
        if file.split('.')[0].endswith(suff):
            os.remove(src+'/'+file)


def main():
    preprocess("./", .8, do_augment=False)


if __name__ == "__main__":
    main()
