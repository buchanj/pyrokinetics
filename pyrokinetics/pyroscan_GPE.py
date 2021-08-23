from .pyro import Pyro
from .pyroscan import PyroScan, set_in_dict, get_from_dict
import os
import numpy as np
from .docker import *
from .gpe_csv import *

# A derived pyroscan class for handling latin hypercube and sequential
# design studies. These each have their own class derived from this.

class PyroScan_GPE(PyroScan):
    """
    A PyroScan derived class for creating and running 
    GPE design cases (LHDs and sequential design studies).

    Member data
    cores_per_run        : How many cores each gs2 run will use
    current_pyro_objects : Pyro objects read from current batch of runs
    all_pyro_objects     : Pyro objects from all runs (used in sequential design)
    gpe_param_dict       : Dictionary of parameters and ranges
    """

    def __init__(self, 
                 pyro,
                 directory = './'
                 image_name = 'gs2_local',
                 template_file = None,
                 param_dict = None,
                 p_prime_type = 0,
                 value_fmt = '.2f',
                 value_separator = '_',
                 parameter_separator = '/',
                 file_name = None,
                 load_default_parameter_keys = True,
                 cores_per_run = 1):
        """
        Initialises some variables
        """

        super().__init__(pyro,
                         param_dict=param_dict,
                         p_prime_type=p_prime_type,
                         value_fmt=value_fmt,
                         value_separator=value_separator,
                         parameter_separator=parameter_separator,
                         file_name=file_name,
                         load_default_parameter_keys=load_default_parameter_keys)

        # Top level run directory
        self.directory = directory

        # Docker image name
        self.image_name = image_name

        # Template input file to use
        self.template_file = template_file

        # Cores to use for each GS2 run
        self.cores_per_run = cores_per_run

        # Current batch of runs
        self.current_pyro_objects = None

        # List of all pyro objects - not just current batch. 
        self.all_pyro_objects  = None

        # Dictionary containing variable names and parameter ranges
        self.gpe_param_dict = None

        # Parameter and target names
        self.parameter_names  = list(self.param_dict.keys())
        self.target_names     = ['mode_frequency','growth_rate']

    def get_parameter_and_target_names():
        """
        Returns parameter and target names.
        """
        return self.parameter_names, self.target_names

    def get_parameter_ranges(self):
        """
        Extracts paramater ranges from pyroscan object.
        """

        self.gpe_param_dict = {}
        
        # Give each parameter an index to ensure correct ordering
        counter = 0
        for param, values in self.param_dict.items():
            
            vmin = np.amin(values)
            vmax = np.amax(values)
            
            self.gpe_param_dict[param] = {}
            self.gpe_param_dict[param][ 'id'] = counter
            self.gpe_param_dict[param]['min'] = vmin 
            self.gpe_param_dict[param]['max'] = vmax
            self.gpe_param_dict[param]['rng'] = vmax - vmin

            counter = counter + 1

    def scale_parameters(self,parameters):
        """
        Takes a set of values in the range [0:1] and scales them
        based on the actual parameter ranges.
        parameters should be a 2D numpy array (nruns,nparams)
        """

        scaled_parameters = []

        for run in range( parameters.shape[0] ):

            params = []
            for param, vdict in self.gpe_param_dict.items():

                # Get index for this parameter
                index = vdict['id']

                # Get value for this iteration
                value = vdict['min'] + vdict['rng'] * parameters[run][index]

                if( value > vdict['max'] or value < vdict['min'] ):
                    raise ValueError(f'Parameter {param} has a value which is out of range: {value}')

                params.append( value )
                
            scaled_parameters.append(params)

        return np.array(scaled_parameters)

    def write_batch(self, parameters, directory='.'):
        """
        Creates and writes GK input files for a set of parameters (numpy array).
        Parameters is expected to be a 2D numpy array (nruns,nparams) containing
        values between 0 and 1 which will be scaled by the actual parameter ranges
        using the above functions.
        """

        if self.gpe_param_dict is None:
            self.get_parameter_ranges()

        nruns = parameters.shape[0]
        print('Submitting '+str(nruns)+' GS2 runs.')

        # Check if parameters are in viable options
        for key in self.param_dict.keys():
            if key not in self.pyro_keys.keys():
                raise ValueError(f'Key {key} has not been loaded into pyro_keys')

        # Scale parameters based on ranges
        scaled_parameters = self.scale_parameters(parameters)
                
        # Iterate through all runs and write output
        for run in range(nruns):

            # Create file name for each run
            run_directory = directory + os.sep + 'iteration_'+str(run) + os.sep

            for param, vdict in self.gpe_param_dict.items():

                # Get index for this parameter
                index = vdict['id']

                # Get attribute and keys where param is stored
                attr_name, keys_to_param, = self.pyro_keys[param]

                # Get dictionary storing the parameter
                param_dict = getattr(self.pyro, attr_name)

                # Set the value given the dictionary and location of parameter
                set_in_dict(param_dict, keys_to_param, scaled_parameters[run,index] )

            self.pyro.write_gk_file(self.file_name, directory=run_directory, template_file=self.template_file)

    def collate_results(self, directory, nruns, wait=True):
        """
        Appends data from completed runs stored in <directory>
        into a set of pyro objects. nruns is the number of 
        runs to read.
        """

        if wait:
            wait_until_finished(self.image)

        self.current_pyro_objects = []

        if self.all_pyro_objects is None:
            self.all_pyro_objects = []

        # Iterate through all runs and recover output
        for run in range(nruns):

            # Directory name for this particular run
            run_directory =  directory + os.sep + 'iteration_'+str(run) + os.sep
            
            # Input file name
            run_input_file = os.path.join(run_directory, self.file_name)
            print('Reading '+run_input_file+' into Pyro object')

            # Read input file into a Pyro object
            pyro = Pyro(gk_file=run_input_file, gk_type='GS2')

            # Read output data
            pyro.gk_code.load_grids(pyro)
            pyro.gk_code.load_fields(pyro)
            pyro.gk_code.load_eigenvalues(pyro)

            # Add this pyro object to lists
            self.current_pyro_objects.append(pyro)
            self.all_pyro_objects.append(pyro)

    def get_parameters_and_targets(self, pyro_objects):
        """
        Returns an array of the varied input parameter and output values (frequency and growth rate)
        for the stored pyro objects contained in pyro_objects.
        """

        inputs  = []
        outputs = []

        # Iterate through all runs and recover output
        for pyro in pyro_objects:

            # Input data for this run
            inputs_ = []

            # Get varied parameter data
            for param in self.param_dict.keys():

                # Get attribute and keys where param is stored
                attr_name, key_to_param, = self.pyro_keys[param]

                # Get dictionary storing the parameter
                param_dict = getattr(pyro, attr_name)

                # Get the required value given the dictionary and location of parameter
                value = get_from_dict(param_dict, key_to_param)

                inputs_.append(value)

            # Get frequency and growth rate
            output_data = pyro.gk_output.data

            frequency   = output_data['mode_frequency']
            growth_rate = output_data['growth_rate']

            # FIXME - probably want some final time averaging here!
            # Create a separate function for extracting frequency and growth rate.
            outputs_ = []
            outputs_.append( np.real( frequency.isel(  time=-1).data[0] ) )
            outputs_.append( np.real( growth_rate.isel(time=-1).data[0] ) )

            inputs.append( inputs_ )
            outputs.append( outputs_ )

        # Store current parameter and target values
        self.parameters = np.array(inputs)
        self.targets    = np.array(outputs)

        return self.parameters, self.targets
        
    def create_csv(self,pyro_objects,directory,filename):
        """
        Creates a CSV file containing the varied parameter data and resulting growth rates.
        This is stored in <directory> and named <filename>
        """

        # Get data
        parameters, targets = self.get_parameters_and_targets(pyro_objects)

        # Write the data file
        create_csv(directory, filename, self.parameter_names, parameters, self.target_names, targets)

    def run(self,directory,nruns,max_containers):
        """ 
        Submits a set of containerised GS2 runs prepared in <directory>.
        """

        # Submit container when cores are available
        run_docker_local(directory,nruns,self.cores_per_run,self.image_name,max_containers)
