import os
import random
import shutil

# town01 short
# town02 tiny


random.seed(2023)

root_path = os.getcwd() 
weather_list = ["weather-%d"%(i) for i in range(21)]
weather_list = ["weather-%d"%(i) for i in [0,2,3,5]]
town_list = ['town01','town02','town03','town04','town05','town06','town07','town10']
frames_cnt = 0
file_list = []
file_name = './index1.txt'

for w in weather_list:
    
    w_path = root_path + '/' + w + '/data/'
    cur_w_list = os.listdir(w_path)
    cur_w_file = [[] for t in range(len(town_list))]
    
    for f in cur_w_list:
        f_path = w_path + f
        f_num = len(os.listdir(f_path + '/rgb_front/'))
        f_res = f_path + "/ %d"%(f_num)
        if f_num > 10:
            for t in range(len(town_list)):
                if town_list[t] in f_res:
                    cur_w_file[t].append(f_res)
        else:
            pass
            #shutil.rmtree(f_path) 
        
    for t in range(len(town_list)):
        file_list += random.sample(cur_w_file[t], min(len(cur_w_file[t]), 100))
    
    
all=[]
with open(file_name, 'w') as f:
    for x in file_list:
        f.writelines(x + '\n')
        cur_cnt=int(x[x.find(' '):])
        all.append([x, cur_cnt])
        frames_cnt += cur_cnt

for i in range(len(all)):
    x=all[i][0]
    wx=0
    townx=0
    longx=0
    cntx=x[x.find(' '):]
    for w in weather_list:
        if w in x:
            wx=chr(int(w[8:]))
    for town in town_list:
        if town in x:
            townx=town
    for long in ['tiny','short','long']:
        if long in x:
            longx=long
            
    all[i][1]=(wx+townx+longx+cntx)
    
all=sorted(all, key=lambda x: x[1])

last=0
for x in all:
    xx=x[0]
    cur_p=xx[:xx.find(' ')]
    cur_n=int(xx[xx.find(' '):])
    longx=0
    for long in ['tiny','short','long']:
        if long in xx:
            longx=long
    if (abs(cur_n-last) < 3 and longx!='tiny') or (abs(cur_n-last) < 1 and longx=='tiny'):
        pass
    last=cur_n
    print(xx)
                


        
print('Frames:', frames_cnt)
        
    
