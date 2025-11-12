import math
import numpy as np
import pandas as pd





def calculate_sample():
    data = pd.read_csv('new_gait_dataset/only_feature_vaild.csv')
    column_means =data.mean()
    class_means = data.groupby('0').mean()
    class_var = data.var()
    each_classnumber=data.groupby('0').size()
    each_classnumber=np.array(each_classnumber)

    sample=np.array([])


    for i in range(8):

        number=0

        for j in range(13):

            x=class_var[j]
            k_mean=class_means.iloc[i, j]
            cl_mean=column_means[j]

            n= math.pow(1.96,2)*x/math.pow((k_mean-cl_mean),2)
            x=each_classnumber[1]

            if n>each_classnumber[i]:
                n=each_classnumber[i]
            number=n+number
        number=math.ceil(number/13)
        sample=np.append(sample,number)
    rows_to_delete = {i + 51: int(sample[i] )for i in range(len(sample))}

    print(rows_to_delete)

    for key, value in rows_to_delete.items():
        data = data.drop(data[data['0'] == key].sample(value).index)
        # filtered_df.to_csv('new_dataset/sample_train_gait_dataset.csv', index=False)
    each_classnumber = data.groupby('0').size()
    data.to_csv('new_gait_dataset/both_original_vaild_dataset.csv', index=False)
    print(each_classnumber)

def reduce_sample(data,rows_to_delete):
    for key, value in rows_to_delete.items():


        data = data.drop(data[data['0'] == key].sample(value).index)
    # filtered_df.to_csv('new_dataset/sample_train_gait_dataset.csv', index=False)
    each_classnumber = data.groupby('0').size()
    # data.to_csv('new_gait_dataset/only_sample_original_gait_dataset.csv', index=False)
    print(each_classnumber)
    # print(filtered_df)






if __name__ == '__main__':
    # stand()


    rows_to_delete=calculate_sample()
    # reduce_sample(data,rows_to_delete)



