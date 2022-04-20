import numpy as np
import torch

from algorithm.clustering import cluster
from algorithm.trainSVM import train
from dao.dao import Dao
from utils.general import cfg
from utils.coordtransform import transform_XYZ2BLH
import datetime
import os


class API:
    def __init__(self):
        self.db = Dao()

    def clustering(self, country='中国', airport='阎良机场',tCode=4010, type='侦察机', model='运-9电子侦察机'):
        self.cluster = cluster
        self.train = train
        table = ""
        if tCode == 4010:
            table = "T_CM_WJZCJ"
        elif tCode == 4009:
            table = "T_CM_WJZHYJJ"
        elif tCode == 4012:
            table == "T_CM_WJZDGJJ"
        elif tCode == 4019:
            table == "T_CM_WJHZJ"
        elif tCode == 4089:
            table == "T_CM_WJMHJ"

        name = '{}-{}-{}'.format(country, airport, model)
        track_line_len = len(self.db.getTrackLineName(country, airport, model))
        if track_line_len:
            for i in range(track_line_len):
                self.db.deleteTrackLinePointsByName(name+'-特征轨迹'+str(i))
            self.db.deleteTrackLineName(country, airport, model)

        # get data from dataset

        country_code = int(self.db.selectByName(country))
        targetID = self.db.selectIDByModelTable(table, country_code, model)
        data = self.db.getHistoryTrackByModel(country_code, airport, tCode, model)
        data = np.array(data)
        targetid = np.array(list(set([i[3] for i in data])))
        b = []
        l = []
        h = []
        for t in targetid:
            tmp = data[data[:, 3] == t]
            xy_tmp = np.stack([tmp[:, 0].astype(np.float32),
                              tmp[:, 1].astype(np.float32)], axis=-1)
            xy_tmp = torch.nn.functional.interpolate(torch.Tensor(
                xy_tmp).unsqueeze(0).transpose(1, 2), size=(cfg.INPUT_LENGHT), mode='linear', align_corners=True).transpose(1, 2).numpy()[0]

            b.append(xy_tmp[:, 0].tolist())
            l.append(xy_tmp[:, 1].tolist())
            h.append(tmp[:, 2].astype(np.float32).tolist())

        # b = None
        # l = None
        # h = None
        train_trajectory = self.cluster([b, l, h])
        avg_trajectory_b, avg_trajectory_l = transform_XYZ2BLH(
            train_trajectory[0], train_trajectory[1], train_trajectory[1])
        train(train_trajectory[2], name)
        linenames = []
        # 未知参数用-1暂时代替
        for i in range(len(avg_trajectory_b)):
            line_time = datetime.datetime.now().strftime(
                '%Y%m%d%H%M%S%f')
            line_import_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S')
            print(line_import_time)
            line_name = name+'-特征轨迹'+str(i)
            self.db.insertTrackLine(ID=line_time,
                                    TargetID=targetID,
                                    TargetSort=tCode,
                                    Name=line_name,
                                    TakeoffAirPortID=airport,
                                    LandingAirPortID=-1,
                                    TaskActionAreaID=-1,
                                    SavePath=os.path.join(
                                        cfg.model_path, name+'.pkl'),
                                    ImportTime=line_import_time)
            linenames.append(line_name)
            for j in range(len(avg_trajectory_b[i])):
                point_time = datetime.datetime.now().strftime(
                    '%Y%m%d%H%M%S%f')
                point_import_time = datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S')
                print(point_import_time)
                point_name = line_name+'-轨迹点'+str(j)
                self.db.insertTrackLinePoint(ID=point_time,
                                             AirLineID=line_time,
                                             Name=point_name,
                                             AirLinePointNo=j,
                                             Longitude=avg_trajectory_l[i][j],
                                             Latitude=avg_trajectory_b[i][j],
                                             Height=-1,
                                             ImportTime=point_import_time)
        return linenames
    def getAllCountryName(self):
        countryCode = self.db.selectAllCountry()
        countryName = []
        for c in countryCode:
            countryName.append(self.db.selectByCode(c))
        return countryName
    def getAirportByCountryName(self,name):
        code = self.db.selectByName(name)
        return self.db.selectStartPlace(code)
    def getTargetType(self,name,airport):
        code = self.db.selectByName(name)
        return self.db.selectTargetType(code,airport)
    def getTargetModel(self,name,airport,targetType):
        code = self.db.selectByName(name)
        return self.db.selectTargetModel(code,airport,targetType)
    def getRT(self,name,airport,type):
        return self.db.getTrackLineName(name,airport,type)
    def getHT(self,name,airport,type,model):
        code = self.db.selectByName(name)
        if model != None:
            return self.db.getHistoryTrackByModel(code,airport,type,model)
        else:
            if type != None:
                return self.db.getHistoryTrackByType(code, airport, type)
            else:
                return self.db.getHistoryTrackByStartPlace(code, airport)

    def getRTPoint(self, name):
        return self.db.getTrackLinePointsByName(name)








