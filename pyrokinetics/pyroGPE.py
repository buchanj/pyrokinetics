
from .pyro import Pyro
from .pyroscan import PyroScan, set_in_dict, get_from_dict
import os
import multiprocessing
import time
import numpy as np
import scipy
import csv
import mogp_emulator as mogp

# Note move csv functions into separate class which pyroscan_LHD and this use - functionality should be together. 
# Likewise make a get_data function for LHD or something which returns varied parameter values and outputs

class PyroGPE:
    """
    A class for generating Gaussian Process Emulators based on
    csv files containing data or pyroscan_LHD objects
    """

    def save(self):
        """
        Save this GPE.
        """

    def load(self):
        """
        Load an already trained GPE.
        """

    def build_from_csv(self,csvfile,kernel='Matern52',nugget=1.0e-8,n_outputs=2):
        """
        Build a Gaussian Process using a csv file containing input parameters
        and growth rate data. By default the last two entries are assumed 
        to be frequency and growth rate respectively.
        """
        
        if kernel not in ['Matern52','SquaredExponential']:
            raise Exception(f'Invalid kernel {kernel} selected.')

        self.csv    = csv
        self.kernel = kernel
        self.nugget = nugget

        # Parameter values at training locations and target values at
        # these locations in parameter space

        self.parameter_values  = []
        self.target_values     = []
        
        with open(csvfile) as cfile:
            
            csvreader = csv.reader(cfile, delimiter=',')

            # Assume first line is row of parameter names
            headers = next(csvreader)
            
            if len(headers) < n_outputs+1:
                raise Exception(f'CSV file contains fewer than {n_outputs} entries per row.')

            self.parameter_names = headers[:-1*n_outputs]
            self.output_names    = headers[-1*n_outputs:]

            for row in csvreader:
                self.parameter_values.append(row[:-1*n_outputs])
                self.target_values.append(row[-1*n_outputs:])

        # Train Gaussian Process 
        self.GPE = mogp.fit_GP_MAP( self.parameter_values, self.target_values, kernel=self.kernel, nugget=self.nugget)

    def build_from_lhd(self,lhd,,kernel='Matern52',nugget=1.0e-8,n_outputs=2):
        """
        Build a Gaussian Process using an existing pyroscan_LHD object containing
        a set of pyro objects.
        """
        
        if kernel not in ['Matern52','SquaredExponential']:
            raise Exception(f'Invalid kernel {kernel} selected.')

        self.csv    = csv
        self.kernel = kernel
        self.nugget = nugget

        # Parameter values at training locations and target values at
        # these locations in parameter space

        self.parameter_names   = lhd.param_dict.keys()
        self.output_names      = [ 'mode_frequency', 'growth_rate' ]

        self.parameter_values  = []
        self.target_values     = []

        # Loop over pyro objects stored in LHD design
        for pyro in lhd.LHD_pyro_objects:

            parameter_values = []

            # Get varied parameter data
            for param in lhd.param_dict.keys():

                # Get attribute and keys where param is stored
                attr_name, key_to_param, = lhd.pyro_keys[param]
            
                # Get dictionary storing the parameter
                param_dict = getattr(pyro, attr_name)
                
                # Get the required value given the dictionary and location of parameter
                value = get_from_dict(param_dict, key_to_param)
            
                parameter_values.append(value)
                
            self.parameter_values.append( parameter_values )

            # Get frequency and growth rate
            output_data = pyro.gk_output.data

            frequency   = output_data['mode_frequency']
            growth_rate = output_data['growth_rate']
            
            outputs = []
            outputs.append( np.real( frequency.isel(  time=-1).data[0] ) )
            outputs.append( np.real( growth_rate.isel(time=-1).data[0] ) )

            self.target_values.append( outputs )

        # Train Gaussian Process 
        self.GPE = mogp.fit_GP_MAP( self.parameter_values, self.target_values, kernel=self.kernel, nugget=self.nugget)

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
