import numpy as np
import datetime
'''     
        This method used to transform a set of points to a set of 
        distance between each points.
======================================================================
        in file form  :

        col1  : x
        col2  : y
        col3  : label of point ; only exist in test dataset.
======================================================================
        out file form:

        col1  : element1
        col2  : element2
        col3  : distance between element1 and element2

'''



def cal_distance(filein):
    print '''
        cal_distance V1.0 
        Finished in Apr 17 12:40 am by LCY
        '''

    now = datetime.datetime.now()
    print now, ":  Start"


    # Points' coordinae
    location = []
    # Points' label ; a variate for test dateset
    label = []


    for line in open(filein, "r"):
        items = line.strip("\n").split(",")
        label.append(int(items.pop()))
        tmp = []
        for item in items:
            tmp.append(float(item))
        location.append(tmp)
    location = np.array(location)
    label = np.array(label)
    length = len(location)

    fileout = filein.split('.')[0] + '.dat'

    fp = open(fileout,'w')

    begin = 0 
    while begin < length -1 :
        end = begin + 1
        while end < length :
            write2File = ''
            dis = np.linalg.norm(location[begin]-location[end])
            write2File = write2File + str(begin+1) + ' ' + str(end + 1) + ' '+ str(dis) + '\n'
            fp.write(write2File)

            end = end + 1
        begin = begin + 1

    fp.close()
    now = datetime.datetime.now()

    print now, ":  Finished"

if __name__ == '__main__' :
    filein = raw_input("Enter dataset : ")
    cal_distance(filein)


