import os
import numpy as np

dir = os.path.dirname(__file__)

class LoadData:

    def animals(self):
        ''' Returns animal data and animal names, (32,84), (32,1) '''

        animal_data, animal_names = [], []

        ''' load animal data '''
        animal_data_file = os.path.join(dir, '../data_lab2/animals.dat')
        with open(animal_data_file) as f:
            data = [int(s) for s in f.readlines()[0].rstrip('\n').split(',')]
            animal_data = np.array(np.array_split(np.array(data),32)) # 32 animals = 84 features per animal

        '''  load animal names '''
        animal_names_file = os.path.join(dir, '../data_lab2/animalnames.txt')
        with open(animal_names_file) as f:
            animal_names = np.array([name.rstrip('\t\n').replace("'","") for name in f.readlines()]).reshape((32,1))

        return animal_data, animal_names

    ''' reads cities file, with x,y coordinates. returns (10,2)'''
    def cities(self):
        cities_path= os.path.join(dir, '../data_lab2/cities.dat')

        with open(cities_path) as f:
            data = f.readlines()
            data = [s.rstrip('\n;').split(',') for s in data[4:]]

            return np.array([ np.array((float(x),float(y))) for (x,y) in data])

    def mp(self):
        districts = [] # (349, 1) there is 1 - 29 districts one row is for one MP member 
        names = [] # (349, 1) one name for each parlament
        party = [] # dict (349,) maps each MP index to correspding party

        # (349, 31), one row is one member, one index in a row is a vote in that voting, total votes is 31
        votes = [] 

        mpdistrict = os.path.join(dir, '../data_lab2/mpdistrict.dat')
        with open(mpdistrict) as f:
            districts = np.array([int(d.strip()) for d in f.readlines()]).reshape((349,1))

        mpnames = os.path.join(dir, '../data_lab2/mpnames.txt')
        with open(mpnames, encoding="ISO-8859-1") as f:
            names = np.array([n.strip() for n in f.readlines()]).reshape((349,1))

        # Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
        d = {
            0: 'no party',
            1: 'm',
            2: 'fp',
            3: 's',
            4: 'v',
            5: 'mp',
            6: 'kd',
            7: 'c',
        }

        mppart = os.path.join(dir, '../data_lab2/mpparty.dat')
        with open(mppart) as f:
            lines = f.readlines()
            member_index = np.arange(0,349)
            p = [d[int(p.strip())] for p in lines[3:]]
            party = dict(zip(member_index, p))
             


        mpvotes = os.path.join(dir, '../data_lab2/votes.dat')
        with open(mpvotes) as f:
            votes = np.array([float(v.strip()) for v in f.readlines()[0].split(',')]).reshape((349,31))

        return districts, names, party, votes
