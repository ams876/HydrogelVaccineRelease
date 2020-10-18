import copy

import numpy as np
from numpy import log
from numpy.random import multinomial, random
import random as rnd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

#from src.general.io_handling import write_results, write_val, write_adj


class StochasticVaccineRelease(object):

    def __init__(self, parameters):
        # parameters contains the size of the protein (so the size of a single lattice), the dims of the lattice,
        # the number of vaccine particles

        self.parameters = parameters

        self.numSubvols1D = self.parameters['numSubvols']
        self.numSubvols = self.parameters['numSubvols'] * self.parameters['numSubvols']
        self.numVaccine = self.parameters['numVaccine']

        self.vaccinePos = rnd.sample(range(0, self.numSubvols), self.numVaccine) #randomly assign each vaccine a different starting position
        self.subvolState = np.ones(self.numSubvols) #track subvolume state, 1 = D_low, 0 = D_high, all start as low D

        self.prop = self.define_propensity()

    def get_adjacent_subvolumes(self, subvolume):
        #-1 = release!

        width = self.numSubvols1D
        adjacent = []

        # 1st: LEFT ADJACENT VOLUME
        # if subvolume modulo width == 0, this selects for column 0
        if (subvolume - 1) >= 0 and (subvolume) % width != 0:
            adjacent.append(subvolume - 1)
        else:
            # if its in column 0, will be released
            adjacent.append(-1)

        # 2nd: RIGHT ADJACENT VOLUME
        if (subvolume + 1) % width != 0:
            adjacent.append(subvolume + 1)
        else:
            # if its in the last column, will be released
            adjacent.append(-1)

        # 3rd: BELOW
        if (subvolume + width) < self.numSubvols:
            adjacent.append(subvolume + width)
        else:
            # if it's in last row, will be released
            adjacent.append(-1)

        # 4th: ABOVE
        if subvolume - width >= 0:
            adjacent.append(subvolume - width)
        else:
            # if it's in first row, will be released
            adjacent.append(-1)

        return adjacent

    # Defines propensity for all reactions and diffusion events
    def define_propensity(self):

        # for each subvolume, the propensity of switching = the state of the subvolume (since 1 if low) * the rate of switching
        subvolSwitching_prop = self.subvolState * self.parameters['k_convert']

        # get prop for each Vac to diffuse
        occupiedSubvols = [i for i, x in enumerate(self.vaccinePos)]
        vacDif_prop = []
        adjacent_open_subvols = []
        for vac_subvolume in self.vaccinePos:

            # find adjacent open subvols
            adjacent = self.get_adjacent_subvolumes(vac_subvolume)
            emptyAdjacent = [i for i in adjacent if i not in occupiedSubvols]

            if self.subvolState[vac_subvolume] == 0:
                # high diffusion state
                # get prop
                prop = self.parameters['D_high'] * len(emptyAdjacent)
                vacDif_prop.append(prop)
                adjacent_open_subvols.append(emptyAdjacent)

            else:
                #low diffusion state
                # get prop
                prop = self.parameters['D_low'] * len(emptyAdjacent)
                vacDif_prop.append(prop)
                adjacent_open_subvols.append(emptyAdjacent)

        propensity = list(subvolSwitching_prop) + vacDif_prop

        return propensity, adjacent_open_subvols

    def run(self, tmax=100, file_num=1):
        tvec = range(0, tmax)

        res = []
        steps, time_array, numRelease = self.GSSA(res, tmax=tmax)
        #write_results(res, file_num)

        return res, time_array, numRelease
        # np.savetxt("time_array_{0}".format(i), time_array, fmt='%f')

        np.savetxt("time", tvec, fmt='%f')
        # np.savetxt("steps", [steps], fmt='%f')

    def GSSA(self, res, tmax=50):
        '''
        Gillespie Direct algorithm
        '''
        tc = 0  # current time
        steps = 0

        # save original state to outputs
        time_array = [0]
        numRelease = [0]

        print("------")
        for tim in range(1, tmax):
            while tc < tim:

                #print('CURRENT TIME = ' + str(tc))

                # calculate propensities for every reaction based on current ini
                pv, adjacent_open_subvols = self.define_propensity()
                a0 = np.sum(pv)  # sum of all transition probabilities
                if a0 == 0:  # no reactions possible -- break
                    break
                # calculate time step based on total propensity
                tau = (-1 / a0) * log(random())

                # choose event
                event = multinomial(1, pv / a0).nonzero()[0][0]

                # we've chosen a subvolume, switch the subvol to high D
                if event < self.numSubvols:
                    self.subvolState[event] = 0 # 0 = high D

                # we've chosen vaccine molecule to diffuse
                else:
                    chosenVac = event - self.numSubvols
                    adjSubvols = adjacent_open_subvols[chosenVac]

                    chosenSubvol = adjSubvols[np.random.randint(len(adjSubvols))]
                    self.vaccinePos[chosenVac] = chosenSubvol

                tc += tau
                steps += 1

            print(self.vaccinePos)
            time_array.append(tim)
            numRelease.append(len([i for i in self.vaccinePos if i == -1]))

        return steps, time_array, numRelease
