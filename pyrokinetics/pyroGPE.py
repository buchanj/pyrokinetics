
from .pyro import Pyro
from .pyroscan import PyroScan, set_in_dict, get_from_dict
import os
import multiprocessing
import time
import numpy as np
import scipy
import csv
import mogp_emulator as mogp

class PyroGPE:
    """
    A class for generating Gaussian Process Emulators based on
    csv files containing data or pyroscan_LHD objects
    """

    def save_gpes(self):
        """
        Save this GPE.
        """

    def load_gpes(self):
        """
        Load an already trained GPE.
        """

    def train_gpes(self,kernel='Matern52',nugget=1.0e-8):
        """
        Builds Gaussian Process Emulators for frequency and
        growth rate based on stored data.
        """

        if kernel not in ['Matern52','SquaredExponential']:
            raise Exception(f'Invalid kernel {kernel} selected.')

        self.kernel = kernel
        self.nugget = nugget

        # Need to reshape target data into separate
        # frequency and growth rate arrays

        targets = np.array(self.target_values)
        targets = targets.transpose()

        # Train Gaussian Processes
        self.frequency_GPE  = mogp.fit_GP_MAP( self.parameter_values, targets[0], kernel=self.kernel, nugget=self.nugget)
        self.growthrate_GPE = mogp.fit_GP_MAP( self.parameter_values, targets[1], kernel=self.kernel, nugget=self.nugget)

    def load_data_from_csv(self,csvfile,n_outputs=2):
        """
        Build a Gaussian Process using a csv file containing input parameters
        and growth rate data. By default the last two entries are assumed 
        to be frequency and growth rate respectively.
        """

        # Store file name
        self.csv = csvfile

        # Parameter values at training locations and target 
        # values at these locations in parameter space

        self.parameter_values  = []
        self.target_values     = []
        
        with open(csvfile) as cfile:
            
            csvreader = csv.reader(cfile, delimiter=',')

            # Assume first line is row of parameter names
            headers = next(csvreader)
            
            if len(headers) < n_outputs+1:
                raise Exception(f'CSV file contains fewer than {n_outputs+1} entries per row.')

            self.parameter_names = headers[:-1*n_outputs]
            self.output_names    = headers[-1*n_outputs:]

            for row in csvreader:
                self.parameter_values.append(row[:-1*n_outputs])
                self.target_values.append(row[-1*n_outputs:])

    def load_data_from_lhd(self,lhd):
        """
        Build a Gaussian Process using an existing pyroscan_LHD object containing
        a set of pyro objects.
        """

        # Get data
        input_names, output_names, inputs, outputs = lhd.get_parameters_and_targets()

        self.parameter_names   = input_names
        self.output_names      = output_names
        self.parameter_values  = inputs
        self.target_values     = outputs

    def validate_from_lhd(self,lhd):
        """
        Perform a validation study using a Latin Hypercube of test data.
        """

    def validate_from_csv(self,csv):
        """
        Perform a validation study using a csv file containing test data.
        """

    def leave_one_out_cross_validate(self):
        """
        Perform a leave one out cross validation study
        """
        
    def evaluate(self,inputs,variance=False):
        """
        Evaluate the GPE at a given input location.
        """
