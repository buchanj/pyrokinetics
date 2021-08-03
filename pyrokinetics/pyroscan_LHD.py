# Add routines for post-analysis. 

import pyroscan
import os
import multiprocessing
import time

class PyroLHS(PyroScan):
    """
    A PyroScan derived class for creating and running 
    Latin Hypercube Scans.
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

    def write(self, npoints=100, file_name=None, directory='.'):
        """
        Creates and writes GK input files for parameters in a maximin Latin Hypercube of size npoints
        """

        from pyDOE import lhs

        self.run_directory = directory

        print(f'Creating a Latin Hypercube of runs with {npoints} points')
        self.latin_hypercube_n = npoints

        if file_name is not None:
            self.file_name = file_name

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

            self.pyro.write_gk_file(self.file_name, directory=run_directory)


    def count_active_containers(self,image_name):
        """
        Counts the number of currently running containers of the given image
        """

        command = 'docker ps | grep ' + image_name + ' | wc -l'
        count = os.system(command)
        
        return count

    def submit_container(self,image_name,run_directory):
        """
        Submits a container of the given image name in the specified run directory
        Assumes the container is set up to awaken in /tmp/work_dir as for the VVeb.UQ app
        """

        command = 'docker run -v ' + run_directory + ':/tmp/work_dir '+ image_name

        print('Submitting container in directory ' + run_directory)
        os.system(command)
 
    def run_docker_local(self,image_name,max_containers):

        """ 
        Submits a set of containerised GS2 runs generated according to a Latin Hypercube Design.
        Currently assumes each run is a single core run.
        """

        # Get total number of available processors
        total_procs = multiprocessing.cpu_count()

        # Iterate through all runs and submit output
        for run in range(self.latin_hypercube_n):

            # Directory name for this particular run
            run_directory = self.run_directory + os.sep + 'LHS_iteration_'+str(run) + os.sep

            while( True ):

                try:

                    active_containers = self.count_active_containers(image_name)
                    
                    # Check if there are processors available for new container
                    if( active_containers < total_procs and
                        active_containers < max_containers ):
                        
                        self.submit_container(image_name,run_directory)
                        
                    else:
                        
                        time.sleep(10)

                except:
                    
                    print('Error submitting container in directory '+run_directory)
                    break
