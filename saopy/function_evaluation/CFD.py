# ==================================================
# author:luojiajie
# ==================================================

import os
import shutil
import numpy as np
import csv

class CFD():
    def __init__(self,rootDir):
        self.rootDir = rootDir # root path


    def calculate(self):
        self.gen_model() # generate CATIA models
        self.gen_mesh() # generate pointwise meshes
        self.CFD()
        # to be continued
        # return y


    def gen_model(self, runtime=300):
        """
        :param runtime: estimated run time (in seconds) for '.catvbs', then force to close CATIA
        """
        print('generating models')
        os.chdir(self.rootDir + '\Model') # change root path to where your '.catvbs' is, otherwise '.catvbs' will not run correctly
        shutil.copyfile('../../../X_new.csv', './X_new.csv')
        os.system('start DOE.catvbs') # run CATIA script
        os.system('ping 127.0.0.1 -n '+str(runtime)+' >nul') # pause runtime seconds
        os.system('taskkill /f /t /im CNEXT.exe') # close CATIA

    # def gen_mesh(self, runtime=20):
    #     """
    #     :param runtime: estimated run time (in seconds) for '.glf', then force to close Pointwise
    #     """
    #     print('generating meshes')
    #     os.chdir(self.rootDir + '\Mesh') # change root path to where your '.glf' is, otherwise '.glf' will not run correctly
    #     os.system('start mesh_script.glf') # run Pointwise script
    #     os.system('ping 127.0.0.1 -n '+str(runtime)+' >nul') # pause runtime seconds
    #     os.system('taskkill /f /t /im Pointwise.exe') # close Pointwise

    def gen_mesh(self, runtime=90):
        # generate mesh if there is more than one script to solve trex mesh error
        print('generating meshes')
        os.chdir(self.rootDir + '\Mesh') # change root path to where your '.glf' is, otherwise '.glf' will not run correctly
        for i in range(31):
            os.system('start M1.glf') # run Pointwise script
            os.system('start M2.glf')  # run Pointwise script
            os.system('ping 127.0.0.1 -n '+str(runtime)+' >nul') # pause runtime seconds
            os.system('taskkill /f /t /im Pointwise.exe') # close Pointwise

    # takeoff
    def CFD(self, runtime=10800):
        os.chdir(self.rootDir)
        #copy all mesh files to 'Cal'
        for i in range(1000):
            if(os.path.exists('./Mesh/' + str(i))):
                shutil.copytree('./Mesh/' + str(i), './Cal/' + str(i))
        os.chdir(self.rootDir + '\Cal')
        os.system('start FLAP_LD_parallel.bat')
        os.system('ping 127.0.0.1 -n ' + str(runtime) + ' >nul')  # pause runtime seconds

        os.chdir(self.rootDir)
        LD=[]
        for i in range(1000):
            if(os.path.exists('./Cal/' + str(i))):
                csvfile = open('./Cal/' + str(i) + '/CLCDCM.csv', 'r', newline='')
                reader = csv.reader(csvfile)
                CL = []
                CD = []
                for row in reader:
                    CL.append(row[1])
                    CD.append(row[2])
                if abs(float(CD[1]) / float(CD[2]) - 1) < 0.01:
                    LD.append(float(CL[2]) / float(CD[2]))
                else:
                    LD.append(40)
                csvfile.close()

        LD=np.array(LD)
        LD.resize((LD.shape[0], 1))
        np.savetxt('y0_new.csv', LD, delimiter=',')


        # remove data in X_new that do not exixt model
        os.chdir(self.rootDir)
        X_new = self.read_csv_to_np('./Model/X_new.csv')

        X_new_final = []

        for i in range(X_new.shape[0]):
            if os.path.exists('./Cal/' + str(i)):
                X_new_final.append(X_new[i, :])
        np.savetxt('X_new.csv', np.array(X_new_final), delimiter=',')

    # landing
    # def CFD(self, runtime=14400):
    #     os.chdir(self.rootDir)
    #     # copy all mesh files to 'Cal'
    #     for i in range(1000):
    #         if (os.path.exists('./Mesh/' + str(i))):
    #             shutil.copytree('./Mesh/' + str(i), './Cal/' + str(i))
    #     os.chdir(self.rootDir + '\Cal')
    #     os.system('start FLAP_clmax_parallel.bat')
    #     os.system('ping 127.0.0.1 -n ' + str(runtime) + ' >nul')  # pause runtime seconds
    #     os.system('start FLAP_clmax_output2.bat')
    #     os.system('ping 127.0.0.1 -n ' + str(180) + ' >nul')  # pause runtime seconds
    #
    #     os.chdir(self.rootDir)
    #     clmax = []
    #     for i in range(1000):
    #         if (os.path.exists('./Cal/' + str(i))):
    #             CLCDCM = self.read_csv_to_np('./Cal/' + str(i) + '/CLCDCM2.csv')
    #             clmax.append(CLCDCM[:, 1].max())
    #
    #     clmax = np.array(clmax)
    #     clmax.resize((clmax.shape[0], 1))
    #     np.savetxt('y0_new.csv', clmax, delimiter=',')
    #
    #     # remove data in X_new that do not exixt model
    #     os.chdir(self.rootDir)
    #     X_new = self.read_csv_to_np('./Model/X_new.csv')
    #
    #     X_new_final = []
    #
    #     for i in range(X_new.shape[0]):
    #         if os.path.exists('./Mesh/' + str(i)):
    #             X_new_final.append(X_new[i, :])
    #     np.savetxt('X_new.csv', np.array(X_new_final), delimiter=',')


    def read_csv_to_np(self,file_name):
        """
        read data from .csv and convert to numpy.array
        note: if you use np.loadtxt(), when the data have only one row or one column, the loaded data is not in the desired shape
        """
        csv_reader = csv.reader(open(file_name))
        csv_data = []
        for row in csv_reader:
            csv_data.append(row)

        row_num = len(csv_data)
        col_num = len(csv_data[0])

        data = np.zeros((row_num, col_num))
        for i in range(row_num):
            data[i, :] = np.array(list(map(float, csv_data[i])))
        return data