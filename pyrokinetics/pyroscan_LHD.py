from .pyro import Pyro
from .pyroscan import PyroScan, set_in_dict, get_from_dict
from .pyroscan_GPE import PyroScan_GPE
import os
import numpy as np

class PyroScan_LHD(PyroScan_GPE):
    """
    A PyroScan derived class for creating and running 
    Latin Hypercube Designs.

    Member data

    run_directory     : Top directory to add individual run directories to         [write]
    latin_hypercube_n : Number of points to generate                               [write]
    file_name         : Name of input file to write                                [write]
    image             : Name of the docker image to be run                         [run]
    """

    def write(self, npoints=100, file_name='gs2.in', directory='.', template_file=None):
        """
        Creates and writes GK input files for parameters in a maximin Latin Hypercube of size npoints
        """

        from pyDOE import lhs

        self.file_name     = file_name
        self.run_directory = directory

        print(f'Creating a Latin Hypercube of runs with {npoints} points')
        self.latin_hypercube_n = npoints

        # Generate a Latin Hypercube
        lhd = lhs(len(self.param_dict), self.latin_hypercube_n, 'maximin')

        super().write_batch(lhd, file_name=file_name, directory=directory, template_file=template_file)

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
        
    def recover_output(self,wait=True):
        """
        Recovers output after runs have completed.
        """
        
        super().collate_results(self.run_directory,self.latin_hypercube_n,filename=self.file_name,wait=wait)

        self.lhd_input_names  = self.parameter_names
        self.lhd_output_names = self.target_names 

        self.lhd_inputs, self.lhd_outputs = super().get_parameters_and_targets(self.current_pyro_objects)

        return self.lhd_input_names, self.lhd_inputs, self.lhd_output_names, self.lhd_outputs

    def run(self,image_name='gs2_local',max_containers=124):
        """
        Checks settings and runs Latin Hypercube Design.
        """

        self.image = image_name
        self.check_settings()

        super().run(self.run_directory,self.latin_hypercube_n,image_name,max_containers)

    def create_csv(self):
        """
        Creates a CSV file containing the varied parameter data and resulting growth rates.
        """

        super().create_csv(self.current_pyro_objects,self.run_directory,self.file_name)

    def submit(self,image_name='gs2_local', npoints=248, directory='./', template_file=None,
               max_containers=124, wait=True):
        """
        Submits the full workflow of designing the LHD, submitting the runs 
        and recovering the output.
        """

        # Generate files
        self.write(npoints=npoints, directory=directory, template_file=template_file)

        # Submit runs
        self.run(image_name=image_name,max_containers=max_containers)
        
        if wait:

            # Recover output data
            input_names, inputs, output_names, outputs = self.recover_output()
            
            # Create csv file
            self.create_csv()
