import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from datetime import timedelta
import random
import argparse
from process_data import process_data


max_cust_one_car = 4
f_name = 'cab_2_hrs.csv'


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
        """
        Initializes a customer object
        :param id_: unique customer id (int)
        :param orig: origin long, lat coordinates (tuple(float, float))
        :param dest: destination long, lat coordinates (tuple(float, float))
        :param tcall: time of request (datetime)
        :param tmin: earliest pickup time (datetime)
        :param fare: fare that was paid by historic passenger (float)
        :param dropoff: @Houming
        :attr served: whether customer has been served (boolean)
        :attr speed: speed of customer at certain time (float)
        """
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

    def __repr__(self):
        return str(self.id)

class arc:
    def __init__(self, cust1, cust2, dist):
        self.cust1 = cust1
        self.cust2 = cust2
        self.dist = dist


class Taxi:
    def __init__(self, id_, pos, time):
        """
        Initializes a Taxi object
        :param id_: unique Taxi id (int)
        :param pos: current position coordinates (tuple(float, float))
        :param time: @Houming
        :attr custs: customers served by taxi (list(int*))
        :attr dropoff: @Houming
        :attr speed: current speed of vehicle
        :attr curr_custs: current customers in taxi (list(int*))
        """
        self.id = id_
        self.pos = pos
        self.time = time
        self.custs = []
        self.dropoff = None
        self.speed = None
        self.curr_custs = []
        self.miles_travelled = 0

    def load(self, c, insert=None):
        """
        Add customer to taxi by adding it to self.custs, and changing the
        position of the taxi to the origin position of the customer. 
        :param c: customer object to add
        :param insert: @Houming
        """
        self.curr_custs.append(c)
        print('Taxi', self.id, 'load Customer', str(c.id), 'at', c.tmin)
        print('Currently in taxi', self.id, ':', [i.id for i in self.curr_custs])
        if insert != None:
            self.custs.insert(insert + 1, c)
        else:
            self.custs.append(c)
        self.miles_travelled += distance(self.pos, c.orig)
        self.pos = c.orig
        self.dropoff = c.dropoff
        c.served = True
        speed = distance(c.dest, c.orig) / (
            c.dropoff - c.tmin - timedelta(seconds=120)
        ).seconds
        # @Houming why do you do this here?
        if speed == 0:
            speed = 0.002
        c.speed = speed
        self.speed = speed

    def unload(self, t):
        """
        @Houming
        """
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
        """
        Indicate whether a taxi can still take more customers by comparing the
        current number of customers with max_cust_one_car
        :returns: boolean
        """
        return False if len(self.curr_custs) >= max_cust_one_car else True

    def insertable(self, t, a):
        """
        @Houming perhaps write out the t and a here since it's not entirely
        clear what they are referring to by just the context alone
        """
        curr_in_car = 0
        new_custs = []
        for i in range(max(a-6, 0), min(a + 6, len(self.custs))):
            if self.custs[i].dropoff > t:
                new_custs.append(self.custs[i])
                curr_in_car += 1
        if max_cust_one_car - 2 <= curr_in_car:
            return False
        self.curr_custs = new_custs
        return True

    def __repr__(self):
        s = ('Taxi ' + str(self.id) + ' at (' + str(self.pos[0]) + ',' +
             str(self.pos[1]) + '):\n')
        for c in self.custs:
            s += ('  Customer ' + str(c.id) + ' from ' + str(c.tmax) +
                  ' to ' + str(c.dropoff) + '\n')
        return s


class RideShareProblem:
    def __init__(self):
        """
        Initialize a RideShareProblem instance
        :param custs: customers to be served (list(class Customer*))
        :param taxis: taxis included in the simulation (list(class Taxi*))
        :param arcs: @Houming
        :param not_assigned: number of customers that were not assigned in the
        ridesharing problem instance
        """
        self.custs = []
        self.taxis = []
        self.arcs = []
        self.not_assigned = num_custs
        self.served_customers = []

    def add_cust(self, customer):
        """
        Add customer to RideShareProblem
        :param customer: class Customer
        """
        self.custs.append(customer)

    def add_taxi(self, taxi):
        """
        Add taxi to RideShareProblem
        :param taxi: class Taxi
        """
        self.taxis.append(taxi)
    
    def add_arc(self, a):
        """
        Add arc to RideShareProblem
        :param arc: class Arc
        """
        self.arcs.append(a)

    def compute_fare(self):
        total_revenue = 0
        for c in self.served_customers:
            total_revenue += c.fare
        return total_revenue

    def solve(self):
        """
        Solve RideSharingProblem naively (offline), by looping through the
        customers, then through the taxis, and add customers where taxis still
        have space
        :return: number of unserved customers
        """
        for c in self.custs:
            dist = np.inf
            take = None
            for t in range(num_taxis):
                # @Houming why do you call unload on the tmax here?
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
<<<<<<< HEAD
=======
                self.served_customers.append(c)
        #for t in self.taxis:
        #    print(t.pos)
>>>>>>> Add information about miles travelled
        print('not assigned: ', self.not_assigned)
        print("Total miles", sum([taxi.miles_travelled for taxi in self.taxis]))
        print(self.compute_fare())


    def nearest(self):
        for c in self.custs:
            dist = np.inf
            take = None
            min_distance = 99999
            for t in range(num_taxis):
                self.taxis[t].unload(c.tmax)
                if self.taxis[t].loadable():
                    dist = distance(self.taxis[t].pos, c.orig)
                    if dist < min_distance:
                        min_distance = dist
                        take = t
                #print(c.id, take)
            if take != None:
                self.not_assigned -= 1
                self.taxis[take].load(c)
                self.served_customers.append(c)
        print('not assigned: ', self.not_assigned)
        print("Total miles", sum([taxi.miles_travelled for taxi in self.taxis]))
        print(self.compute_fare())

                        
    def greedy_heuristic(self):
        """
        Improve the solution of the RideSharingProblem by performing greedy
        heuristic insertion. See our final report for details on the working of
        this algorithm. 
        :return: number of unserved customers
        """
        for c in self.custs:
            if not c.served:
                dist = np.inf
                take = None
                insert = None
                for t in range(num_taxis):
                    for a in range(len(self.taxis[t].custs)-1):
                        # @Houming it would be helpful here to more explicitly
                        # write what these variables are, or to alternatively
                        # add comments about what they are (though the former
                        # is preferred)
                        c_k_1 = self.taxis[t].custs[a]
                        c_k = self.taxis[t].custs[a+1]
                        T_c_ck = distance(c.orig, c_k.orig) / c_k.speed
                        tmin_cs = max(
                            c.tmin, c_k.tmax - timedelta(seconds=T_c_ck)
                        )
                        T_c_ck1 = distance(c_k_1.orig, c.orig) / c_k_1.speed
                        tmax_cs = min(
                            c.tmax, c_k_1.dropoff + timedelta(seconds=T_c_ck1)
                        )
                        tmp_dist = distance(c_k_1.dest, c.orig)
                        if (tmin_cs <= tmax_cs and tmp_dist < dist and
                            self.taxis[t].insertable(tmax_cs, a)):
                            dist = tmp_dist
                            take = t
                            insert = a
                if take != None:
                    self.served_customers.append(c)
                    self.not_assigned -= 1
                    self.taxis[take].load(c, insert)
<<<<<<< HEAD
=======
        #for t in self.taxis:
        #     print(t.pos)
>>>>>>> Add information about miles travelled
        print('not assigned: ', self.not_assigned)
        print(self.compute_fare())
        print("Total miles", sum([taxi.miles_travelled for taxi in self.taxis]))

            

def check_insert(pb, t, i, c):
    c_k_1 = pb.taxis[t].custs[i-1]
    c_k = pb.taxis[t].custs[i]
    T_c_ck = distance(c.orig, c_k.orig) / c_k.speed
    tmin_cs = max(c.tmin, c_k.tmax - timedelta(seconds=T_c_ck))
    T_c_ck1 = distance(c_k_1.orig, c.orig) / c_k_1.speed
    tmax_cs = min(c.tmax, c_k_1.dropoff + timedelta(seconds=T_c_ck1))
    return tmin_cs <= tmax_cs and pb.taxis[t].insertable(tmax_cs, i)

def opt(pb):
    """
    @Houming
    """
    not_assigned = pb.not_assigned
    for t in range(num_taxis):
        new_cust = [pb.taxis[t].custs[0]]
        i = 1
        num = len(pb.taxis[t].custs)
        while i < num:
            for t1 in range(t + 1, num_taxis):
                new_cust1 = [pb.taxis[t1].custs[0]]
                i1 = 1
                run = True
                num1 = len(pb.taxis[t1].custs)
                while i1 < num1 and run:
                    c_i_1 = pb.taxis[t].custs[i]
                    c_i1 = pb.taxis[t1].custs[i1]
                    tmin_new = c_i_1.tmin
                    T_c_ci1 = distance(c_i_1.dest, c_i1.orig) / c_i_1.speed
                    if tmin_new + timedelta(seconds=T_c_ci1) <= c_i1.tmax:
                        tmp = list(new_cust)
                        tmp1 = list(new_cust1)
                        for j in range(i, num):
                            tmp1.append(pb.taxis[t].custs[j])
                        for j in range(i1, num1):
                            tmp.append(pb.taxis[t1].custs[j])
                        pb.taxis[t].custs = tmp
                        pb.taxis[t1].custs = tmp1
                        for c in pb.custs:
                            if not c.served:
                                if check_insert(pb, t, i, c):
                                    pb.taxis[t].load(c, i-1)
                                    new_cust.append(c)
                                    pb.not_assigned -= 1
                                    i += 1
                                elif check_insert(pb, t1, i1, c):
                                    pb.taxis[t1].load(c, i1-1)
                                    new_cust1.append(c)
                                    pb.not_assigned -= 1
                                    i1 += 1
                        num = len(pb.taxis[t].custs)
                        num1 = len(pb.taxis[t1].custs)
                        run = False
                    new_cust1.append(pb.taxis[t1].custs[i1])
                    i1 += 1
            new_cust.append(pb.taxis[t].custs[i])
            i += 1
    print(pb.not_assigned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--taxis',
        type=int,
        default=100,
        help="Number of taxis active in simulation"
    )
    parser.add_argument(
        '--customers',
        type=int,
        default=1000,
        help="Number of customers within time window"
    )
    args = parser.parse_args()
    num_taxis = args.taxis
    num_custs = args.customers

    # read and filter data
    df = process_data(f_name)
    random.seed(2)
    sample = sorted(random.sample(range(df.shape[0]), num_custs))

    # create the taxi problem
    pb = RideShareProblem()
    for i, s in enumerate(sample):
        row = df.iloc[s]
        orig = (row['pickup_latitude'], row['pickup_longitude'])
        dest = (row['dropoff_latitude'], row['dropoff_longitude'])
        t = row['tpep_pickup_datetime']
        # make assumption on tcall, tmin, tmax
        tcall = timedelta(
            hours=int(t[11:13]),
            minutes=int(t[14:16])-3,
            seconds=int(t[17:19])
        )
        tmin = timedelta(
            hours=int(t[11:13]),
            minutes=int(t[14:16])-2,
            seconds=int(t[17:19])
        )
        tmax = timedelta(
            hours=int(t[11:13]),
            minutes=int(t[14:16])+2,
            seconds=int(t[17:19])
        )
        t = row['tpep_dropoff_datetime']
        dropoff = timedelta(
            hours=int(t[11:13]),
            minutes=int(t[14:16]),
            seconds=int(t[17:19])
        )
        pb.add_cust(
            Customer(
                i+1, orig, dest, tcall, tmin, tmax, float(row['fare_amount']),
                dropoff
            )
        )

    for index, row in df.head(num_taxis).iterrows():
        start_pos = (row['pickup_latitude'], row['pickup_longitude'])
        print(start_pos[0])
        pb.add_taxi(Taxi(index, start_pos, row['tpep_pickup_datetime']))
    for index, row in df.head(num_taxis).iterrows():
        start_pos = (row['pickup_latitude'], row['pickup_longitude'])
        print(start_pos[1])

    custs = pb.custs
    for i in range(len(custs)):
        dest = custs[i].dest
        for j in range(i + 1, len(custs)):
            orig = custs[j].orig
            pb.add_arc(arc(custs[i], custs[j], distance(dest, orig)))

    
    #pb.greedy_heuristic()
    pb.solve()
    pb.greedy_heuristic()
    opt(pb)
    #pb.solve()
    #pb.nearest()
