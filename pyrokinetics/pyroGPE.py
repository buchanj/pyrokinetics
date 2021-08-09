
from .pyro import Pyro
from .pyroscan import PyroScan, set_in_dict, get_from_dict
import os
import multiprocessing
import time
import numpy as np
import scipy
import csv
import mogp_emulator as mogp

# FIXME - Convert to using Xarrays. 

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

        if self.parameter_values is None or self.target_values is None:
            raise Exception("No training data availble in GPE class.")

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

        if self.frequency_GPE is None or self.growthrate_GPE is None:
            raise Exception("Cannot perform validation as GPE is not trained.")

        if lhd.lhd_inputs is None:
            lhd.get_parameters_and_targets()

        freq, gamma, freq_unc, gamma_unc = self.evaluate(lhd.lhd_inputs,uncertainty=True)

        # Calculate Z values = (value - mean)/sd
        frequency_zs = np.zeros(lhd.latin_hypercube_n)
        gamma_zs     = np.zeros(lhd.latin_hypercube_n)

        frequency_rms_error = 0.0
        gamma_rms_error     = 0.0

        for i in range( lhd.latin_hypercube_n ):
            
            
            frequency_target = lhd.lhd_outputs[i,0]
            gamma_target     = lhd.lhd_outputs[i,1]

            frequency_prediction = freq[i]
            gamma_prediction     = gamma[i]

            frequency_uncertainty = freq_unc[i]
            gamma_uncertainty     = gamma_unc[i]

            frequency_zs[i] = ( frequency_prediction - frequency_target ) / frequency_uncertainty 
            gamma_zs[i]     = (     gamma_prediction -     gamma_target ) / gamma_uncertainty 

            frequency_rms_error = frequency_rms_error + ( frequency_prediction - frequency_target )**2.0
            gamma_rms_error     = gamma_rms_error     + (     gamma_prediction -     gamma_target )**2.0

        frequency_rms_error = frequency_rms_error**0.5 / lhd.latin_hypercube_n
        gamma_rms_error     =     gamma_rms_error**0.5 / lhd.latin_hypercube_n

        # Scale by range of data 

        # Get minima and maxima of frequency and growth rate over test data
        minima = np.amin( lhd.lhd_outputs, 0 )
        maxima = np.amax( lhd.lhd_outputs, 0 )

        frequency_rms_error = frequency_rms_error / ( maxima[0] - minima[0] )
        gamma_rms_error     =     gamma_rms_error / ( maxima[1] - minima[1] )

    def validate_from_csv(self,csv):
        """
        Perform a validation study using a csv file containing test data.
        """

        # Read into LHD and then call above. 

    def leave_one_out_cross_validate(self):
        """
        Perform a leave one out cross validation study
        """
        # To Do.
        
    def evaluate(self,inputs,uncertainty=False):
        """
        Evaluate the GPE at a given input location.
        Input is an Npoints * Ninputs array of parameter values, outputs
        are Npoints sized arrays of predictions and uncertainties.
        """

        try:
            
            freq, freq_unc   = self.frequency_GPE.predict(inputs,unc=uncertainty)
            gamma, gamma_unc = self.growthrate_GPE(inputs,unc=uncertainty)

        else:

            raise Exception("Error evaluating Gaussian Process Emulators")

        return freq, gamma, freq_unc, gamma_unc
        
