import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from datetime import timedelta
import random

num_taxis = 100
num_custs = 1000
wait_time = timedelta(minutes=5) # 5 mins
max_cust_one_car = 4
f_name = 'yellow_tripdata_2016-04.csv'


def distance(cord1, cord2):
    """
    Takes two coordinates, and returns the distance between them
    in km
    :param cord1: Starting coordinate with longitude and latitude value
    :param cord2: End coordinate with longitude and latitude
    :returns: distance bewtween cord1 and cord2 in kilometers
    """
    # approximate radius of earth in km
    R = 6373.0
    
    lat1 = radians(cord1[0])
    lon1 = radians(cord1[1])
    lat2 = radians(cord2[0])
    lon2 = radians(cord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


class Customer:
    def __init__(self, id_, orig, dest, tcall, tmin, tmax, fare, dropoff):
        self.id = id_
        self.orig = orig
        self.dest = dest
        self.tcall = tcall
        self.tmin = tmin
        self.tmax = tmax
        self.fare = fare
        self.dropoff = dropoff
        self.served = False
        self.speed = None


class arc:
    def __init__(self, cust1, cust2, dist):
        self.cust1 = cust1
        self.cust2 = cust2
        self.dist = dist


class Taxi:
    def __init__(self, id_, pos, time):
        self.id = id_
        self.pos = pos
        self.time = time
        self.custs = []
        self.dropoff = None
        self.speed = None
        self.curr_custs = []

    def load(self, c, insert=None):
        self.curr_custs.append(c)
        curr = ''
        print('Taxi', self.id, 'load Customer', str(c.id), 'at', c.tmin)
        for i in self.curr_custs:
            curr += str(i.id) + ' '
        print('Currently in taxi', self.id, ':', curr)
        if insert:
            self.custs.insert(insert + 1, c)
        else:
            self.custs.append(c)
        self.pos = c.orig
        self.dropoff = c.dropoff
        c.served = True
        speed = distance(c.dest, c.orig) / (c.dropoff - c.tmin - timedelta(seconds=120)).seconds
        if speed == 0:
            speed = 0.002
        c.speed = speed
        self.speed = speed

    def unload(self, t):
        new_custs = []
        curr = ''
        for c in self.curr_custs:
            if t < c.dropoff:
                new_custs.append(c)
                curr += str(c.id) + ' '
            else:
                print('Taxi', self.id, 'unload Customer', str(c.id), 'at', c.dropoff)
                print('Currently in taxi', self.id, ':', curr)
        self.curr_custs = new_custs
    
    def loadable(self):
        if len(self.curr_custs) >= max_cust_one_car:
            return False
        return True
    
    def __repr__(self):
        s = 'Taxi ' + str(self.id) + ' at (' + str(self.pos[0]) + ',' + str(self.pos[1]) + '):\n'
        for c in self.custs:
            s += '  Customer ' + str(c.id) + ' from ' + str(c.tmax) + ' to ' + str(c.dropoff) + '\n'
        return s


class TaxiProblem:
    def __init__(self):
        self.custs = []
        self.taxis = []
        self.arcs = []
        self.not_assigned = num_custs

    def add_cust(self, customer):
        self.custs.append(customer)

    def add_taxi(self, taxi):
        self.taxis.append(taxi)
    
    def add_arc(self, a):
        self.arcs.append(a)

    def solve(self):
        for c in self.custs:
            dist = np.inf
            take = None
            for t in range(num_taxis):
                self.taxis[t].unload(c.tmax)
                if self.taxis[t].loadable():
                    tmp_dist = distance(self.taxis[t].pos, c.orig)
                    if self.taxis[t].speed != None:
                        T = tmp_dist / self.taxis[t].speed
                    else:
                        T = 0
                    if tmp_dist < dist and (c.tcall + timedelta(seconds=T)) < c.tmax:
                        dist = tmp_dist
                        take = t
            if take != None:
                self.not_assigned -= 1
                self.taxis[take].load(c)
        # for t in self.taxis:
        #     print(t)
        print('not assigned: ', self.not_assigned)

    def greedy_heuristic(self):
        self.solve()
        x = True
        for c in self.custs:
            if not c.served:
                dist = np.inf
                take = None
                insert = None
                for t in range(num_taxis):
                    for a in range(len(self.taxis[t].custs)-1):
                        c_k_1 = self.taxis[t].custs[a]
                        c_k = self.taxis[t].custs[a+1]
                        T_c_ck = distance(c.dest, c_k.orig) / c_k.speed
                        tmin_cs = max(c.tmin, c_k.tmax - timedelta(seconds=T_c_ck))
                        T_c_ck1 = distance(c_k_1.dest, c.orig) / c_k_1.speed
                        tmax_cs = min(c.tmax, c_k_1.tmin + timedelta(seconds=T_c_ck1))
                        tmp_dist = distance(c_k_1.dest, c.orig)
                        if tmin_cs <= tmax_cs and tmp_dist < dist and self.taxis[t].loadable():
                            if x:
                                x = False
                            dist = tmp_dist
                            take = t
                            insert = a
                if take != None:
                    self.not_assigned -= 1
                    self.taxis[take].load(c, insert)
        # for t in self.taxis:
        #     print(t)
        print('not assigned: ', self.not_assigned)


if __name__ == "__main__":
    # read and filter data
    df = pd.read_csv(f_name, nrows=10000)
    relevant_columns = [
        'tpep_pickup_datetime', 
        'tpep_dropoff_datetime',
        'trip_distance',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'fare_amount'
    ]
    df = df[relevant_columns]

    df = df[df.trip_distance < 100]
    df = df[df.trip_distance > 0.]
    df = df[df.pickup_longitude != 0]
    df = df[df.dropoff_longitude != 0]
    df = df[df.tpep_pickup_datetime > '2016-04-02']
    df = df[df.tpep_pickup_datetime < '2016-04-05']
    df = df.reset_index()
    random.seed(2)
    sample = sorted(random.sample(range(df.shape[0]), num_custs))

    # create the taxi problem
    pb = TaxiProblem()
    for i, s in enumerate(sample):
        row = df.iloc[s]
        orig = (row['pickup_latitude'], row['pickup_longitude'])
        dest = (row['dropoff_latitude'], row['dropoff_longitude'])
        t = row['tpep_pickup_datetime']
        # make assumption on tcall, tmin, tmax
        tcall = timedelta(hours=int(t[11:13]), minutes=int(t[14:16])-3, seconds=int(t[17:19]))
        tmin = timedelta(hours=int(t[11:13]), minutes=int(t[14:16])-2, seconds=int(t[17:19]))
        tmax = timedelta(hours=int(t[11:13]), minutes=int(t[14:16])+2, seconds=int(t[17:19]))
        t = row['tpep_dropoff_datetime']
        dropoff = timedelta(hours=int(t[11:13]), minutes=int(t[14:16]), seconds=int(t[17:19]))
        pb.add_cust(Customer(i+1, orig, dest, tcall, tmin, tmax, float(row['fare_amount']), dropoff))

    for index, row in df.head(num_taxis).iterrows():
        start_pos = (row['pickup_latitude'], row['pickup_longitude'])
        pb.add_taxi(Taxi(index, start_pos, row['tpep_pickup_datetime']))

    custs = pb.custs
    for i in range(len(custs)):
        dest = custs[i].dest
        for j in range(i + 1, len(custs)):
            orig = custs[j].orig
            pb.add_arc(arc(custs[i], custs[j], distance(dest, orig)))
    
    # pb.greedy_heuristic()
    pb.solve()
