import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dataScreening(data, name):
    """Screens inpit data for 0 detercted enegry sum after removing true enegry
        column:

        Takes:
            data(pandas dataframe object) = energy data from each layer, true
                                            energy in first column. Detected
                                            energies in columns >0. Assumed no
                                            data other than numeric is inside the df
            name(string) = name of the particle or data set to print out item
                            indicies removed.

        Returns:
            data(pandas dataframe object) = the OG dataframe but with no zero
                                                sum detected energy values

    """
    badIndicies = []
    newDf = data.drop('True', axis = 1)
    for i in range(data.shape[0]):
        sumVal = sum(newDf.iloc[i].values)
        if sumVal < (0.05 * data.iloc[i]['True']):
            print('Data Point: {} Removed for having zero detected energy :: {}'.format(i, name))
            badIndicies.append(i)
    data = data.reset_index()
    if len(badIndicies) > 0:
        badIndicies.reverse()
        for i in range(len(badIndicies)):
            data = data.drop(badIndicies[i], axis = 0)
    data = data.set_index('index')
    return data

def calibration(data):
    """Perfroms calibartion of energy data values using the summated detected
        energies and the true energy values:

        Takes:
            data(pandas dataframe object) = energy data from each layer, true
                                            energy in first column. Detected
                                            energies in columns >0. Assumed no
                                            data other than numeric is inside the df

        Returns:
            data(pandas dataframe object) = a new df with additonal columns for
                                            calibration ratio, calibrated energy
                                            and summated detected energy
    """
    true = data['True']
    summable = data.drop('True', axis = 1)
    newColumn = summable.sum(axis = 1).values
    data['Esum'] = newColumn
    meanVal = np.mean((data['True']/data['Esum']).values)
    ratios = true/newColumn
    data['Ratios'] = ratios
    data['Ecal'] = meanVal * newColumn
    data['Calibrations'] = (data['Ecal'] - true)/true
    return data

def calibration1d(data, name, energy, ax, nbins = 50, ranges = [-1, 1]):
    """Plots a simple hostogram of 'Calibrations' values:

        Takes:
            data(pandas dataframe object) = only requires a 'Calibrations' column
            name(string) = the name of teh particle used in the simulations
            energy(int or String)(in MeV) =energy at which the simulations took place
            ax(matplotlib object) = the canvas/axis to plot this histogram on
            nbins(int) = number of bins to use in the histogram
            ranges(size 2 array) = element 0 is lower bound and element 1 is upper bound

        Returns:
            None

        Outputs:
            graph(histogram) = binned calibration data for a given dataframe
    """
    print('Resoltuion: {:.3f} for {} at {} MeV'.format(np.std(data['Calibrations']), name, energy))
    ax.hist(data['Calibrations'], bins = nbins, range = ranges)
    ax.set_xlabel('(Calibrated energy- true energy)/true energy')
    ax.set_ylabel('Counts')
    ax.set_title('Calibration Plot for {} at {} MeV'.format(name, energy))

def calibration2d(data, name, ax, nbins = (10, 20)):
    """Plots a 2d histogram of enegry values and their calibration value:

        Takes:
            data(pandas dataframe object) = only requires a 'Calibrations' column
            name(string) = the name of teh particle used in the simulations
            ax(matplotlib object) = the canvas/axis to plot this histogram on
            nbins(size 2 tuple) = element 0 is bins along x, element 1 is bins across y

        Returns:
            None

        Outputs:
            graph(2d histogram) = binned calibration data for a given dataframe
    """
    ax.hist2d(data['True'] , data['Calibrations'] , bins = nbins )
    ax.set_xlabel('True Energy [MeV]')
    ax.set_ylabel('(Calibrated energy- true energy)/true energy')
    ax.set_title('Calibration Plot for {}'.format(name))
