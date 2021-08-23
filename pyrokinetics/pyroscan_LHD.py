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

    def write(self, npoints=100):
        """
        Creates and writes GK input files for parameters in a maximin Latin Hypercube of size npoints
        """

        from pyDOE import lhs

        print(f'Creating a Latin Hypercube of runs with {npoints} points')
        self.latin_hypercube_n = npoints

        # Generate a Latin Hypercube
        lhd = lhs(len(self.param_dict), self.latin_hypercube_n, 'maximin')

        super().write_batch(lhd, directory=self.directory)

    def check_settings(self):
        """
        Checks data needed for post-processing steps is available.
        """

        # Check LHD size is set
        if self.latin_hypercube_n is None:
            raise Exception('No LHD information available. Aborting.')

    def run(self,max_containers=124):
        """
        Checks settings and runs Latin Hypercube Design.
        """
        self.check_settings()

        super().run(self.directory,self.latin_hypercube_n,max_containers)

    def create_csv(self,filename):
        """
        Creates a CSV file containing the varied parameter data and resulting growth rates.
        """

        super().create_csv(self.current_pyro_objects,self.directory,filename)

    def submit(self,npoints=248, max_containers=124, filename='LHD.csv', wait=True):
        """
        Submits the full workflow of designing the LHD, submitting the runs 
        and recovering the output.
        """

        # Generate files
        self.write(npoints=npoints)

        # Submit runs
        self.run(max_containers=max_containers)
        
        if wait:

            # Recover output data
            inputs, output = self.get_parameters_and_targets(self.current_pyro_objects)

            # Create csv file
            self.create_csv(filename)
