
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

    def create_gpes(self,parameters,targets,kernel='Matern52',nugget=1.0e-8):
        """
        Build Gaussian Process Emulators for frequency and growth rate based
        on training data.
        
        parameters is [n_points,n_parameters]
        targets is [n_points,2] where the 2 outputs are frequency and growth rate
        """

        if kernel not in ['Matern52','SquaredExponential']:
            raise Exception(f'Invalid kernel {kernel} selected.')

        # Reshape target data into separate frequency and growth rate arrays.
        targets_ = targets.transpose()

        # Train Gaussian Processes
        frequency_GPE  = mogp.fit_GP_MAP( parameters, targets_[0], kernel=kernel, nugget=nugget)
        growthrate_GPE = mogp.fit_GP_MAP( parameters, targets_[1], kernel=kernel, nugget=nugget)

        return frequency_GPE, growthrate_GPE

    def evaluate_gpes(self,inputs,frequency_GPE,growthrate_GPE,uncertainty=True):
        """
        Evaluate the GPEs at a given input location.
        Input is an [n_points,n_parameters] array of parameter values, outputs
        are n_points sized arrays of predictions and uncertainties.
        """

        try:
            
            freq, freq_unc   = frequency_GPE.predict(inputs,unc=uncertainty)
            gamma, gamma_unc = growthrate_GPE.predict(inputs,unc=uncertainty)

        except:

            raise Exception("Error evaluating Gaussian Process Emulators")

        return freq, gamma, freq_unc, gamma_unc

    def train(self,kernel='Matern52',nugget=1.0e-8):
        """
        A wrapper to the above create_gpes function for training the main Gaussian 
        Processes based on the training data (as opposed to cross validation cases).
        """
        
        if self.parameter_values is None or self.target_values is None:
            raise Exception("No training data available to train GPEs")

        # Store kernel and nugget
        self.kernel = kernel
        self.nugget = nugget

        self.frequency_GPE, self.growthrate_GPE = \
            self.create_gpes(self.parameter_values,self.target_values,kernel=kernel,nugget=nugget)

    def predict(self,inputs,uncertainty=True):
        """
        Likewise a wrapper to evaluate_gpes for the main Gaussian Processes
        """

        if self.frequency_GPE is None or self.growthrate_GPE is None:
            raise Exception("GPEs have not been trained. Cannot make predictions.")

        return self.evaluate_gpes(inputs,self.frequency_GPE,self.growthrate_GPE,uncertainty=uncertainty)
     
    # Tools for loading training and test data -------------------------------------

    def load_training_data_from_lhd(self,lhd):
        """
        Stores the data from the LHD class into member data of the GPE class to
        be used for training.
        """

        # Get data
        if lhd.lhd_inputs is None:
            lhd.get_parameters_and_targets()

        self.parameter_names   = lhd.lhd_input_names
        self.target_names      = lhd.lhd_output_names
        self.parameter_values  = lhd.lhd_inputs
        self.target_values     = lhd.lhd_outputs
        self.training_n        = len( self.parameter_names )

    def read_csv(self,csvfile,n_outputs=2):
        """
        Read data from a csv file containing input parameters and growth 
        rate data. By default the last two entries are assumed to be 
        frequency and growth rate respectively.
        """

        # Parameter values at training locations and target 
        # values at these locations in parameter space

        parameter_values  = []
        target_values     = []
        
        with open(csvfile) as cfile:
            
            csvreader = csv.reader(cfile, delimiter=',')

            # Assume first line is row of parameter names
            headers = next(csvreader)
            
            if len(headers) < n_outputs+1:
                raise Exception(f'CSV file contains fewer than {n_outputs+1} entries per row.')

            parameter_names = headers[:-1*n_outputs]
            output_names    = headers[-1*n_outputs:]

            for row in csvreader:
                parameter_values.append(row[:-1*n_outputs])
                target_values.append(row[-1*n_outputs:])

        return parameter_names, parameter_values, target_names, target_values

    def load_training_data_from_csv(self,csvfile,n_outputs=2):
        """
        Wrapper to the above data for training data. Stores the data from
        the CSV file into member data of the GPE class
        """

        # Store file name
        self.csv = csvfile

        # Names and values of inputs and outputs
        self.parameter_names, self.parameter_values, self.target_names, self.target_values = self.read_csv(csvfile,n_outputs=n_outputs)

        self.training_n = len( self.parameter_names )

    # Validation tools ----------------------------------------------------------

    def get_Z_values(self,mean,sd,value):
        """
        Returns standardised deviation of a value based on the 
        mean and standard deviation.
        """

        if sd != 0:
            z = (value-mean) / sd
        else:
            z = 0
            
        return z

    def validate_against_test_data(self,inputs,targets):
        """
        Perform a validation study against test data.
        inputs is [npoints,nparameters] while targets is [npoints,2]
        """
        
        assert inputs.shape[0] == targets.shape[0]
        ntest = inputs.shape[0]

        if self.frequency_GPE is None or self.growthrate_GPE is None:
            raise Exception("Cannot perform validation as GPE is not trained.")

        # Evaluate Gaussian Processes at test locations
        frequency_prediction, gamma_prediction, frequency_uncertainty, gamma_uncertainty = self.predict(inputs)

        # Z values = (value - mean)/sd
        frequency_zs = np.zeros( ntest )
        gamma_zs     = np.zeros( ntest )

        frequency_rms_error = 0.0
        gamma_rms_error     = 0.0

        # Loop over points and calculate squared difference between target and prediction
        for i in range( ntest ):
            
            frequency_target = targets[i,0]
            gamma_target     = targets[i,1]

            frequency_zs[i] = self.get_Z_values(frequency_prediction[i],frequency_uncertainty[i],frequency_target)
            gamma_zs[i]     = self.get_Z_values(    gamma_prediction[i],    gamma_uncertainty[i],    gamma_target)

            frequency_rms_error = frequency_rms_error + ( frequency_prediction[i] - frequency_target )**2.0
            gamma_rms_error     = gamma_rms_error     + (     gamma_prediction[i] -     gamma_target )**2.0

        # Take square roots to get RMS and normalise by number of data items
        frequency_rms_error = frequency_rms_error**0.5 / ntest
        gamma_rms_error     =     gamma_rms_error**0.5 / ntest

        # Scale by range of data 

        # Get minima and maxima of frequency and growth rate over test data
        minima = np.amin( targets, 0 )
        maxima = np.amax( targets, 0 )

        frequency_rms_error = frequency_rms_error / ( maxima[0] - minima[0] )
        gamma_rms_error     =     gamma_rms_error / ( maxima[1] - minima[1] )

        return frequency_rms_error, gamma_rms_error, frequency_zs, gamma_zs   

    def validate_from_lhd(self,lhd):
        """
        Perform a validation study using a Latin Hypercube of test data.
        """

        if lhd.lhd_inputs is None:
            lhd.get_parameters_and_targets()

        if self.parameter_names != lhd.lhd_input_names or self.target_names != lhd.lhd_output_names:
            raise Exception("Test data does not match training data!")

        return self.validate_against_test_data(lhd.lhd_inputs,lhd.lhd_outputs)

    def validate_from_csv(self,csvfile,n_outputs=2):
        """
        Perform a validation study using data stored in a csv file
        """

        # Names and values of inputs and outputs
        parameter_names, parameter_values, target_names, target_values = self.read_csv(csvfile,n_outputs=n_outputs)

        if self.parameter_names != parameter_names or self.target_names != target_names:
            raise Exception("Test data does not match training data!")

        return self.validate_against_test_data(parameter_values,target_values)

    def leave_one_out_cross_validate(self):
        """
        Perform a leave one out cross validation study
        """

        # Calculate Z values = (value - mean)/sd
        frequency_zs = np.zeros( self.training_n )
        gamma_zs     = np.zeros( self.training_n )

        frequency_rms_error = 0.0
        gamma_rms_error     = 0.0

        # Loop over the training data and retrain omitting one data point
        for i in range( self.training_n ):

            # Create a new parameter set omitting the current data item
            params  = np.copy(self.parameter_values)
            targets = np.copy(self.target_values)

            # Store values to be omitted
            loocv_params = params[i]
            loocv_target = targets[i]

            # Remove these from training array
            np.delete(params,i)
            np.delete(targets,i)

            # Retrain a Gaussian Process using the reduced data set
            frequency_GPE, growthrate_GPE = self.create_gpes(params,targets,kernel=self.kernel,nugget=self.nugget)

            # Compare the prediction with the omitted value
            freq, gamma, freq_unc, gamma_unc = self.evaluate_gpes(loocv_params,frequency_GPE,growthrate_GPE)

            # Calculate deviations
            frequency_zs[i] = self.get_Z_values( freq, freq_unc,loocv_target[0])
            gamma_zs[i]     = self.get_Z_values(gamma,gamma_unc,loocv_target[1])

            frequency_rms_error = frequency_rms_error + (  freq - loocv_target[0] )**2.0
            gamma_rms_error     = gamma_rms_error     + ( gamma - loocv_target[1] )**2.0

        frequency_rms_error = frequency_rms_error**0.5 / self.training_n
        gamma_rms_error     =     gamma_rms_error**0.5 / self.training_n

        # Scale by range of data 

        # Get minima and maxima of frequency and growth rate over test data
        minima = np.amin( self.target_values, 0 )
        maxima = np.amax( self.target_values, 0 )

        frequency_rms_error = frequency_rms_error / ( maxima[0] - minima[0] )
        gamma_rms_error     =     gamma_rms_error / ( maxima[1] - minima[1] )

        return frequency_rms_error, gamma_rms_error, frequency_zs, gamma_zs