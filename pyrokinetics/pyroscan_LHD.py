# Need to write a run script based on example scan and example analysis
# This should use this to submit a batched LHD scan then once that has
# finished loop over the results directories and aggregate the results.  

from .pyro import Pyro
from .pyroscan import PyroScan, set_in_dict, get_from_dict
import os
import multiprocessing
import time
import numpy as np
import csv

class PyroLHD(PyroScan):
    """
    A PyroScan derived class for creating and running 
    Latin Hypercube Designs.

    Member data
    lhs_param_dict    : Dictionary of parameters and ranges used to generate LHD   [get_parameter_ranges]
    run_directory     : Top directory to add individual run directories to         [write]
    latin_hypercube_n : Number of points to generate                               [write]
    file_name         : Name of input file to write                                [write]
    LHD_pyro_objects  : A collection of pyro objects for each run with output data [collate_results]

    """

    def get_parameter_ranges(self):

        self.lhs_param_dict = {}
        
        # Give each parameter an index to ensure correct ordering
        counter = 0
        for param, values in self.param_dict.items():
            
            vmin = np.amin(values)
            vmax = np.amax(values)
            
            self.lhs_param_dict[param] = {}
            self.lhs_param_dict[param][ 'id'] = counter
            self.lhs_param_dict[param]['min'] = vmin 
            self.lhs_param_dict[param]['max'] = vmax
            self.lhs_param_dict[param]['rng'] = vmax - vmin

            counter = counter + 1

    def write(self, npoints=100, file_name='gs2.in', directory='.', template_file=None):
        """
        Creates and writes GK input files for parameters in a maximin Latin Hypercube of size npoints
        """

        from pyDOE import lhs

        self.file_name     = file_name
        self.run_directory = directory

        print(f'Creating a Latin Hypercube of runs with {npoints} points')
        self.latin_hypercube_n = npoints

        # Check if parameters are in viable options
        for key in self.param_dict.keys():
            if key not in self.pyro_keys.keys():
                raise ValueError(f'Key {key} has not been loaded into pyro_keys')

        # Get parameter ranges
        self.get_parameter_ranges()

        # Generate a Latin Hypercube
        lhs = lhs(len(self.param_dict), self.latin_hypercube_n, 'maximin')

        # Iterate through all runs and write output
        for run in range(self.latin_hypercube_n):

            # Create file name for each run
            run_directory = directory + os.sep + 'LHS_iteration_'+str(run) + os.sep

            for param, vdict in self.lhs_param_dict.items():

                # Get index for this parameter
                index = vdict['id']
                
                # Get value for this iteration
                value = vdict['min'] + vdict['rng'] * lhs[run][index]

                if( value > vdict['max'] or value < vdict['min'] ):
                    raise ValueError(f'Parameter {param} has a value which is out of range: {value}')

                # Get attribute and keys where param is stored
                attr_name, keys_to_param, = self.pyro_keys[param]

                # Get dictionary storing the parameter
                param_dict = getattr(self.pyro, attr_name)

                # Set the value given the dictionary and location of parameter
                set_in_dict(param_dict, keys_to_param, value)

            run_input_file = os.path.join(run_directory, self.file_name)

            self.pyro.write_gk_file(self.file_name, directory=run_directory, template_file=template_file)

    def count_active_containers(self,image_name):
        """
        Counts the number of currently running containers of the given image
        """

        command = 'docker ps | grep ' + image_name + ' | wc -l'
        count = int( os.popen(command).read() )
        
        return count

    def get_absolute_path(self,run_directory):
        """
        Docker won't attach volumes using relative paths so need to convert relative
        paths to absolute ones before submitting containers.
        """
        
        return os.path.abspath(run_directory)

    def submit_container(self,image_name,run_directory):
        """
        Submits a container of the given image name in the specified run directory in detached
        mode. Assumes the container is set up to awaken in /tmp/work_dir as for the VVeb.UQ app
        """

        abs_run_directory = self.get_absolute_path(run_directory)

        command = 'docker run -v ' + abs_run_directory + ':/tmp/work_dir -d '+ image_name
        print('Submitting container in directory ' + abs_run_directory)
        os.system(command)
        print('Submitted')

    def check_settings(self):
        """
        Checks data needed for post-processing steps is available.
        """

        # Check LHD size is set
        if self.latin_hypercube_n is None:
            raise Exception('No LHD information available. Aborting.')

        # Check run directory
        if self.run_directory is None:
            print('Run directory is unset, assuming pwd.')
            self.run_directory = '.'
        
    def run_docker_local(self,image_name,max_containers):

        """ 
        Submits a set of containerised GS2 runs generated according to a Latin Hypercube Design.
        Currently assumes each run is a single core run.
        """

        # Check settings
        self.check_settings() 

        # Get total number of available processors
        total_procs = multiprocessing.cpu_count()

        # Iterate through all runs and submit output
        for run in range(self.latin_hypercube_n):

            # Directory name for this particular run
            run_directory = self.run_directory + os.sep + 'LHS_iteration_'+str(run) + os.sep

            while( True ):

                try:

                    active_containers = self.count_active_containers(image_name)
                    print( str(active_containers) + ' active containers' )

                    # Check if there are processors available for new container
                    if( active_containers < total_procs and
                        active_containers < max_containers ):
                        
                        self.submit_container(image_name,run_directory)
                        break
                        
                    else:
                        
                        print('Waiting for available cores...')
                        time.sleep(10)

                except:
                    
                    print('Error submitting container in directory '+run_directory)
                    break
    
    def collate_results(self):
        """
        Reads data from completed LHD runs into a set of pyro objects.
        """

        # Check settings
        self.check_settings()

        self.LHD_pyro_objects = []

        # Iterate through all runs and recover output
        for run in range(self.latin_hypercube_n):

            # Directory name for this particular run
            run_directory =  self.run_directory + os.sep + 'LHS_iteration_'+str(run) + os.sep
            
            # Input file name
            run_input_file = os.path.join(run_directory, self.file_name)
            print('Reading '+run_input_file+' into Pyro object')

            # Read input file into a Pyro object
            pyro = Pyro(gk_file=run_input_file, gk_type='GS2')

            # Read output data
            pyro.load_gk_output()

            # Add this pyro object to list
            self.LHD_pyro_objects.append(pyro)

    def get_parameters_and_targets(self):
        """
        Returns an array of the varied input parameter and output values (frequency and growth rate)
        for the stored pyro objects created by the Latin Hypercube scan.
        """
        
        # Check results information is stored
        if self.LHD_pyro_objects is None:
            self.collate_results()

        self.lhd_input_names  = list(self.param_dict.keys())
        self.lhd_output_names = ['mode_frequency','growth_rate']

        self.lhd_inputs  = []
        self.lhd_outputs = []

        # Iterate through all runs and recover output
        for pyro in self.LHD_pyro_objects:

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
            outputs_ = []
            outputs_.append( np.real( frequency.isel(  time=-1).data[0] ) )
            outputs_.append( np.real( growth_rate.isel(time=-1).data[0] ) )

            self.lhd_inputs.append( inputs_ )
            self.lhd_outputs.append( outputs_ )

        self.lhd_inputs  = np.array( self.lhd_inputs )
        self.lhd_outputs = np.array( self.lhd_outputs )

        return self.lhd_input_names, self.lhd_inputs, self.lhd_output_names, self.lhd_outputs

    def create_csv(self):
        """
        Creates a CSV file containing the varied parameter data and resulting growth rates
        """

        # Get data
        input_names, inputs, output_names, outputs = self.get_parameters_and_targets()

        with open( self.run_directory + os.sep + 'LHD.csv', 'w' ) as csvfile:

            # CSV writer
            csvwriter = csv.writer(csvfile, delimiter=',')

            # Create a header line containing the varied parameters and growth rates
            headers = input_names + output_names
            csvwriter.writerow(headers)

            # Iterate through all runs and recover output
            for run in range(len(inputs)):
                
                data = np.array( list(inputs[run]) + list(outputs[run]) )
                csvwriter.writerow(data)
