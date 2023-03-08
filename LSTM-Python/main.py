import joblib
from tensorflow import keras
#import tflite_runtime.interpreter as tflite
import pandas as pd
import numpy as np
from time import time as t
import csv
import os
directory=os.path.dirname(os.path.abspath(__file__))
standard_scaler = joblib.load(os.path.join(directory,'standard_scaler.pickle'))

model_number = 23
input_len = 15 #30 detik kebelakang
output_len = 5 #5 detik kedepan
input_file = 'perdetikesp2 test.csv'
output_file = 'output-lstm-esp2.csv'
scala = 1
def Average(lst):
    return sum(lst) / len(lst)

def make_time_step_dataset(data, time_steps, num_output):
    x, y = [], []
    for i in range(len(data) - time_steps - num_output + 1):
        temp = data[i:(i+time_steps), 0]
        x.append(temp)
        temp = data[(i + time_steps):(i+time_steps+num_output), 0]
        y.append(temp)

    return np.array(x), np.array(y)

def readTxtToData(data):
    
    columns = ['throughput','durasi','ntp_time','data_collection_count','block_sequence_length','num_devices','lower_outlier','mac']

    row = {
        columns[0]: [],
        columns[1]: [],
        columns[2]: [],
        columns[3]: [],
        columns[4]: [],
        columns[5]: [],
        columns[6]: [],
        columns[7]: []
    }
    
    #READ FILE A FILE
    a_file = open(data, "r")
    lines = a_file.readlines()
    last_xlines = lines[-(9999):]
    a_file.close()
    #CLOSE FILE
    baris = 0
    for data in last_xlines:
        split = data.split(";")
        row['mac'].append("08:3A:F2:A9:8D:85")
        start = float(split[1]) + (float(split[2])/1000000)
        row['ntp_time'].append(str(start))
        baris = baris + 1
        row['data_collection_count'].append(baris)
        row['block_sequence_length'].append(1)
        end = float(split[3]) + (float(split[4])/1000000)
        row['throughput'].append(round(float(split[0])/(end-start))/scala )
        row['num_devices'].append(1)
        row['lower_outlier'].append(False)
        row['durasi'].append(end-start)
     
    df = pd.DataFrame(row, columns=columns)
    print(df)
    with open("esp1-v1.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ntp_time","throughput","durasi"])
        for index, row in df.iterrows():
            
            writer.writerow([row['ntp_time'], row['throughput'], row['durasi']])
    return df


def loadModel():
    config = joblib.load(os.path.join(directory,f'final_model/{model_number}_model_dump.pickle'), mmap_mode='r')
    model = keras.Sequential().from_config(config)
    model.load_weights(os.path.join(directory,f'final_model/{model_number}.h5'))

    return model

def generate_data_from_dataset(a):
    data_test = readTxtToData(a)
    #print(data_test[0:100])
    #Pre-process data
    columns2 = ['ntp_time','throughput']
    row2 = {
        columns2[0]: [],
        columns2[1]: []   
    }
    start_time = float(data_test.iloc[0]["ntp_time"])
    tp_per_x_sec = 1
    end_time = start_time + tp_per_x_sec
    buffer_durasi = 0
    buffer_th = 0
    for index, row in data_test.iterrows():
        th = row['throughput']
        start = float(row['ntp_time'])
        durasi = row['durasi']

        while start >= end_time:
            tp_updated = (buffer_durasi * buffer_th)
            row2['ntp_time'].append(str(end_time - tp_per_x_sec))
            row2['throughput'].append(tp_updated/tp_per_x_sec)
            buffer_durasi = 0
            end_time = end_time + tp_per_x_sec
            buffer_th = th
        
        if (start + durasi >= end_time): #rekaputulasi data
            
            if (durasi >= tp_per_x_sec):
                buffer_durasi_local = buffer_durasi + durasi
                durasi_round_down = np.floor(buffer_durasi_local)

                for i in range(int(durasi_round_down)):
                    print(i)
                    if (i == 0):
                        tp_updated = (buffer_durasi * buffer_th) + ((1-buffer_durasi)*th)
                        
                        row2['ntp_time'].append(str(end_time - tp_per_x_sec))
                        row2['throughput'].append(tp_updated/tp_per_x_sec)
                        end_time = end_time + tp_per_x_sec
                    else :
                        
                        row2['ntp_time'].append(str(end_time - tp_per_x_sec))
                        row2['throughput'].append(th/tp_per_x_sec)
                        end_time = end_time + tp_per_x_sec
                buffer_durasi = buffer_durasi_local - durasi_round_down 
                buffer_th = th    
                
            else:
                total_durasi = buffer_durasi + durasi
                excess = total_durasi - tp_per_x_sec
                if (excess < 0):
                    th_updated = (buffer_durasi * buffer_th) + ((durasi) * th) 
                    row2['ntp_time'].append(str(end_time - tp_per_x_sec))
                    row2['throughput'].append(th_updated/tp_per_x_sec)
                    end_time = end_time + tp_per_x_sec
                    buffer_durasi = 0
                    buffer_th = 0
                else:
                    th_updated = (buffer_durasi * buffer_th) + ((durasi - excess) * th)    
                    row2['ntp_time'].append(str(end_time - tp_per_x_sec))
                    row2['throughput'].append(th_updated/tp_per_x_sec)
                    end_time = end_time + tp_per_x_sec
                    buffer_durasi = excess
                    buffer_th = th     
    
        else : 
                 
            buffer_th = (buffer_durasi * buffer_th) + (durasi * th)       
            buffer_durasi = buffer_durasi +durasi
            buffer_th = buffer_th / buffer_durasi

            
    preprocessData = pd.DataFrame(row2, columns=columns2)

    return preprocessData

        #print(row['throughput'], row['ntp_time'], row['durasi'])

def process_LSTM_Prediction(data,model):
    
    output_data=[]
    global_time = []
    for index, row in data.iterrows():
        input_data =[]
        if ((index+1) % output_len == 0):
            if (index < input_len-1):
                sample_data = data.iloc[0:index+1]['throughput']             
                time = row['ntp_time']            
                input_data.append(sample_data)            
                total_x = np.array(input_data)           
                total_x =np.concatenate([total_x[0], np.zeros(input_len-index-1)])
                
                #sample_data = sample_data.values.tolist()
                #listofzeros = [[0.0]] * (input_len-index-1)          
                #input_data = [*sample_data, *listofzeros]       
                total_x_valid = []
                total_x_valid.append(total_x)   
                total_x_valid = np.array(total_x_valid)   
                # total_x_valid = np.expand_dims(total_x_valid, axis=0)                 
                pred_y = model.predict(total_x_valid)
                print("ini disni")
                #print(pred_y)
                pred_y = pred_y[0]
                output_data.append([time,Average(pred_y)*scala]) 
                
                
                fc = standard_scaler.inverse_transform([pred_y])    
                print(fc)
                fc = fc[0]
                print(fc)
                print(Average(fc)*scala)
                output_data.append([time,Average(fc)*scala])               
              
            else :
                sample_data = data.iloc[index-(input_len-1):index+1]['throughput']
                #print(sample_data)
                time = row['ntp_time']
                input_data.append(sample_data)
                total_x = np.array(input_data)
                #print(total_x)
                start_time_exe = t()
                print(total_x)
                pred_y = model.predict(total_x)
                pred_y = pred_y[0]
                print(pred_y)
                fc = standard_scaler.inverse_transform([pred_y])    
                fc = fc[0]
                print(Average(fc))
                output_data.append([time,Average(fc)*scala])

                end_time_exe = t()
                global_time.append(end_time_exe - start_time_exe)
                print(end_time_exe-start_time_exe) 
    #print(output_data)
    ave = Average(global_time)
    print("TIME : " + str(ave))
    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time","prediction"])
        for data in output_data:        
            writer.writerow([data[0],data[1]])


# pd.set_option("display.max_rows", None, "dislay.max_columns", None)

model = loadModel()
data = generate_data_from_dataset(input_file)
print(data)

aa = data["throughput"]
bb = np.array(aa)
cc= bb.reshape(-1,1)
standard_scaler.fit(cc)
print(standard_scaler.mean_)
transform = standard_scaler.transform(cc)
for i in range(0,len(cc)):  
    data.loc[i,"throughput"]= transform[i][0]
print(data)

process_LSTM_Prediction(data,model)




# data_test = readTxtToData('dataSheetType2.txt')
# total_x = np.empty((0, 15))
# tr = data_test[['throughput']]
# trArray = tr.to_numpy()
# x, y = make_time_step_dataset(trArray, 15, 5)
# total_x = np.append(total_x, x, axis=0)
# print(total_x)
# total_x = total_x.reshape(-1, 15, 1)
# print(total_x)
# start2 = time.time()
# pred_y = model.predict(total_x)
# print(pred_y)
# end = time.time()
# print("TOTAL DURASI = :" + str(end-start2))