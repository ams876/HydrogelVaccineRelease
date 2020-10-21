import copy

import numpy as np
from numpy import log
from numpy.random import multinomial, random
import random as rnd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import argparse
import os
import matplotlib.pyplot as plt
import math



#from src.general.io_handling import write_results, write_val, write_adj


class StochasticVaccineRelease(object):

    def __init__(self, parameters):
        # parameters contains the size of the protein (so the size of a single lattice), the dims of the lattice,
        # the number of vaccine particles

        self.parameters = parameters

        self.numSubvols1D = self.parameters['numSubvols']
        self.numSubvols = self.parameters['numSubvols'] * self.parameters['numSubvols'] * self.parameters['numSubvols']
        self.numVaccine = self.parameters['numVaccine']

        self.vaccinePos = rnd.sample(range(0, self.numSubvols), self.numVaccine) #randomly assign each vaccine a different starting position
        self.subvolState = np.ones(self.numSubvols) #track subvolume state, 1 = D_low, 0 = D_high, all start as low D

        self.prop = self.define_propensity()

    def get_adjacent_subvolumes(self, subvolume):
        #-1 = release!
        
        #print("sub")
        #print(subvolume)

        width = self.numSubvols1D
     
        face_max= (int(math.ceil(subvolume / (width*width))) * (width*width))
        face_min= face_max-(width*width)
        adjacent = []

        # 1st: LEFT ADJACENT VOLUME
        # if subvolume modulo width == 0, this selects for column 0
        if (subvolume - 1) >= face_min and (subvolume) % width != 0:
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
        if (subvolume + width) < (face_max):
            adjacent.append(subvolume + width)
        else:
            # if it's in last row, will be released
            adjacent.append(-1)

        # 4th: ABOVE
        if subvolume - width >= face_min:
            adjacent.append(subvolume - width)
        else:
            # if it's in first row, will be released
            adjacent.append(-1)
            
        #5th: Middle Front 
        if subvolume - (width*width)>=0:
            adjacent.append(subvolume-(width*width))
        else:
            adjacent.append(-1)
            
        
        #6th: Middle Behind 
        if subvolume + (width*width)<self.numSubvols:
            adjacent.append(subvolume+(width*width))
        else:
            adjacent.append(-1)
                       
        #7th: Top Right
        
        #excludes the last column 
        if subvolume - (width-1) >= face_min & (subvolume) %width != (width-1):
            adjacent.append(subvolume - (width-1))
        else:
            adjacent.append(-1)
            
        #8th: Top Left
        
        #excludes the first column
        
        if subvolume - (width+1) >= face_min & (subvolume) % width != 0:
            adjacent.append(subvolume - width- 1)
        else:
            adjacent.append(-1)
        
        #9th: Bottom Right
        
        if (subvolume + (width+1)) < (face_max) & (subvolume) %width != (width-1):
            adjacent.append(subvolume + width+1)
        else:
            # if it's in last row, will be released
            adjacent.append(-1)
        
        #10th: Bottom Left 
        
        return adjacent
    
        if (subvolume + (width-1)) < (face_max) & (subvolume) % width !=0:
            adjacent.append(subvolume + width-1)
        else:
            # if it's in last row, will be released
            adjacent.append(-1)
        
        #11th: Top Right Front
                       
        #checks to make sure not front most and not right most 
        
        if  face_max>(width*width) & (subvolume) %width != (width-1) & (subvolume - (width*width) - (width-1)) >= (face_min-(width*width)):
            adjacent.append(subvolume - (width*width) - (width-1))
        else:
            adjacent.append(-1)            
                       
        #12th: Top Left Front
        
        #checks to make sure not front most and not left most 
                       
        if  face_max>(width*width) & (subvolume) %width != (0) & (subvolume - (width*width) - (width+1)) >= (face_min-(width*width)):
            adjacent.append(subvolume - (width*width) - (width+1))
        else:
            adjacent.append(-1) 
                     
        #13th: Bottom Right Front
        
        #checks to make sure not front most and not right most 
        
        if  face_max>(width*width) & (subvolume) %width != (width-1) & (subvolume - (width*width) + (width+1)) >= (face_min-(width*width)):
            adjacent.append(subvolume - (width*width) + (width+1))
        else:
            adjacent.append(-1)    
                       
        #14th: Bottom Left Front 
                       
        #checks to make sure not front most and not left most 
        
        if  face_max>(width*width) & (subvolume) %width != 0 & (subvolume - (width*width) - (width-1)) >= (face_min-(width*width)):
            adjacent.append(subvolume - (width*width) - (width-1))
        else:
            adjacent.append(-1)  
                     
                       
        #15th: Top Right Back
        
        #checks to make sure not back most and not right most
                       
        if face_max<(width*width*width) & (subvolume) %width != (width-1) & (subvolume +(width*width) - (width-1)) < (face_max+(width*width)):
            adjacent.append(subvolume +(width*width) - (width-1))
        else:
            adjacent.append(-1)  
                       
        #16th: Top Left Back
        
         #checks to make sure not back most and not left most              
                       
        if face_max<(width*width*width) & (subvolume) %width != 0 & (subvolume +(width*width) - (width+1)) < (face_max+(width*width)):
            adjacent.append(subvolume +(width*width) - (width+1))
        else:
            adjacent.append(-1)  
                     
        #17th: Bottom Right Back
                       
        #checks to make sure not back most and not right most               
                       
        if face_max<(width*width*width) & (subvolume) %width != (width-1) & (subvolume +(width*width) + (width+1)) < (face_max+(width*width)):
            adjacent.append(subvolume +(width*width) + (width+1))
        else:
            adjacent.append(-1) 
                       
        #18th: Bottom Left Back 
        
        #checks to make sure not back most and not left most              
                       
        if face_max<(width*width*width) & (subvolume) %width != 0 & (subvolume +(width*width) + (width-1)) < (face_max+(width*width)):
            adjacent.append(subvolume +(width*width) + (width-1))
        else:
            adjacent.append(-1) 
                       
        #19th: Top Face Middle Back
        
        #check to make sure not back most 
         
        if face_max<(width*width*width) & (subvolume +(width*width) + (width)) < (face_max+(width*width)):
            adjacent.append(subvolume +(width*width) + (width))
        else:
            adjacent.append(-1) 
                       
        
        #20th: Top Face Middle Front
                  
        #check to make sure not front most 
                       
        if  face_max>(width*width) & (subvolume - (width*width) - (width)) >= (face_min-(width*width)):
            adjacent.append(subvolume - (width*width) - (width))
        else:
            adjacent.append(-1) 
        
                       
        #21th: Bottom Face Middle Back
        
        #check to make sure not back most 
         
        if face_max<(width*width*width) & (subvolume +(width*width) - (width)) < (face_max+(width*width)):
            adjacent.append(subvolume +(width*width) - (width))
        else:
            adjacent.append(-1) 
          
        #22th: Bottom Face Middle Front
        #check to make sure not front most 
                       
        if  face_max>(width*width) & (subvolume - (width*width) + (width)) >= (face_min-(width*width)):
            adjacent.append(subvolume - (width*width) + (width))
        else:
            adjacent.append(-1)
                       
        #23th: Middle Face Back Right
        #check to make sure not back most and not right most
         
        if face_max<(width*width*width) & (subvolume) %width != (width-1) & (subvolume +(width*width) +1) < (face_max+(width*width)):
            adjacent.append(subvolume +(width*width) +1)
        else:
            adjacent.append(-1) 
                       
        #24th: Middle Face Back Left 
                       
        #check to make sure not back most and not left most 
         
        if face_max<(width*width*width) & (subvolume) %width != 0 & (subvolume +(width*width) -1) < (face_max+(width*width)):
            adjacent.append(subvolume +(width*width) -1)
        else:
            adjacent.append(-1) 
                       
        #25th: Middle Face Front Right
        
        #check to make sure not front most and not right most 
                       
        if  face_max>(width*width) & (subvolume) %width != (width-1) & (subvolume - (width*width) - 1) >= (face_min-(width*width)):
            adjacent.append(subvolume - (width*width) -1)
        else:
            adjacent.append(-1)
                       
        #26th: Middle Face Front Left and not left most             
         
        if  face_max>(width*width) & (subvolume) %width != 0 & (subvolume - (width*width) + 1) >= (face_min-(width*width)):
            adjacent.append(subvolume - (width*width) + 1)
        else:
            adjacent.append(-1)
                       
                       

    # Defines propensity for all reactions and diffusion events
    def define_propensity(self):

        # for each subvolume, the propensity of switching = the state of the subvolume (since 1 if low) * the rate of switching
        subvolSwitching_prop = self.subvolState * self.parameters['k_convert']

        # get prop for each Vac to diffuse
        #print("numsolvs")
        #print(self.numSubvols)
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

            #print(self.vaccinePos)
            time_array.append(tim)
            numRelease.append(len([i for i in self.vaccinePos if i == -1]))

        return steps, time_array, numRelease
    
class Procedure(object):

    def __init__(self, parameters):
        self.parameters = parameters
        self.home = os.getcwd()

    def run_StochasticVaccineRelease(self, tmax=20, reps=1, file_num=1, ):
        sim = StochasticVaccineRelease(parameters=self.parameters)
        #model = Spatial_LLPS_Model(sim)
        res, time, numRelease = sim.run(tmax=tmax, file_num=file_num)
        return res, time, numRelease


    def get_trajectories(self):
        print("Starting post-processing...")
        out = KP_LAT_Exit(num_odes=self.gillespieLAT.len_ini)
        output_traj = out.output_trajectories
        time_vector = out.time_vector
        # plot_output_trajectories(output_traj, time_vector)
        return output_traj, time_vector


def define_n_initial(R, Ls, La, LckZap, LAT):
    n_initial = np.zeros(9, dtype=int)
    n_initial[0] = R
    n_initial[1] = Ls
    n_initial[2] = La
    n_initial[5] = LckZap
    n_initial[8] = LAT
    return n_initial


def define_parameters():
    # Define all key parameters of the simulation

    numVaccine = 15 # number of vaccine particles
    vaccineSize = 0.05 # the size of a single vaccine particle, used to define our lattice dimension
    numSubvols = 10 # size of the lattice in 1D (so if = 10, there are 10^3 total sites)

    D_high = 5E-3 / vaccineSize**2 # diffusion coef for a vaccine within a high D square
    D_low = 0 / vaccineSize**2  # diffusion coef for a vaccine within a low D square
    k_convert = 0.01  # 1/s, the rate at which a single square converts from low to high D
    delta_k_convert = 0  # 1/s, the rate at which k_convert increases as a function of time? maybe not eh

    parameters = {'D_high': D_high, 'D_low': D_low, 'k_convert': k_convert,
                  'delta_k_convert' : delta_k_convert, 'numVaccine' : numVaccine,
                  'numSubvols': numSubvols, 'vaccineSize': vaccineSize}

    return parameters


if __name__ == "__main__":
    #need to uncomment this and add in other parameters for running from the cluster
    #parser = argparse.ArgumentParser(description="Testing LAT recruitment simulation.",
    #                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #parser.add_argument('--file_num', dest='file_num', action='store', type=int,
    #                    help="Number of the output traj file.")
    #args = parser.parse_args()

    file_num = 0 #args.file_num
    tmax = 10000
    subvols = 20
    parameters = define_parameters()
    print(parameters)
    protocol = Procedure(parameters)
    res, time, numRelease = protocol.run_StochasticVaccineRelease(tmax=tmax, file_num=file_num)

    print(parameters)

    fig = plt.figure()
    plt.plot(time, numRelease)
    plt.xlabel('time')
    plt.ylabel('num vaccine particles released')
    plt.show()




    # visualize traj run locally

    '''output = PostProcess('./trajectory_7', protocol.mapping, tmax)  # ./../outputs/trajectory_3
    traj = output.traj
    print(traj)
    visualize3(
        {k: traj[k] for k in ('R', 'Ls', 'La', 'RLs', 'RLa', 'RLs_LckZap', 'RLa_LckZap', 'RLa_LckZap_LAT', 'LAT')},
        tmax, subvols, (3, 3))
    visualizeLattice(vals, Adj, tmax, subvols)
'''
