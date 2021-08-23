from .pyro import Pyro
from .pyroscan import PyroScan, set_in_dict, get_from_dict
from .pyroscan_GPE import PyroScan_GPE
import os
import numpy as np
import mogp_emulator as mogp

# FIXME - make scaling optional? FIXME - Make sure scaling is handled consistently. 
# Or train a GPE at the end using the unscaled data???
# Basically got as far as writing submission functions for initial LHD and MICE batches
# Need to recover data and handle updating the MICE design.
# Then need some kind of stopping criterion. 
# May want option to test against data between updates and output convergence. 
    
class PyroScan_MICE(PyroScan_GPE):
    """
    A PyroScan derived class for creating and running 
    MICE sequential Designs.

    Member data
    run_directory     : Top directory to add individual run directories to         [write]
    latin_hypercube_n : Number of points to generate for initial LHD               [write]
    lhd               : The LHD object the MICE object is built from               [write]
    file_name         : Name of input file to write                                [write]

    """

    def create_design(self, image_name='gs2_local', n_init=124, n_cand=50, file_name='gs2.in', 
                      directory='.', template_file=None):
        """
        Creates the MICE design object and writes an initial Latin Hypercube Design.
        """

        self.image_name    = image_name
        self.file_name     = file_name
        self.run_directory = directory
        self.template_file = template_file

        print(f'Creating an initial Latin Hypercube run with {n_init} points')
        self.latin_hypercube_n = n_init

        # Generate a Latin Hypercube
        self.lhd = mogp.LatinHypercubeDesign( len(self.param_dict) )

        # Initialise MICE sequential design
        self.mice_design = mogp.MICEDesign( lhd, n_init=n_init, n_cand=n_cand )

        # Initial Design
        lhd = self.mice_design.generate_initial_design()

        # Write files
        lhd_run_directory = directory + os.sep + 'batch_0'
        super().write_batch(lhd, file_name=file_name, directory=lhd_run_directory, template_file=template_file)

    def create_mice_batch(self, batch_number, n_batch=8):
        """
        Uses MICE to generate a batch of runs and writes the input files for them.
        """

        print(f'Creating a MICE batch with {n_batch} points')

        # Generate a batch
        self.n_batch = n_batch
        batch = self.mice_design.get_batch_points( n_batch )

        # Write files
        run_directory = self.run_directory + os.sep + 'batch_' + str(batch_number)
        super().write_batch(batch, file_name=self.file_name, directory=run_directory, template_file=self.template_file)

    def check_settings(self,batch_number):
        """
        Checks data needed for submission and post-processing steps is available.
        """

        # Check batch size is set 
        if batch_number > 0:
            if self.n_batch is None:
                raise Exception('No batch size information available. Aborting.')
        else:
            # Check LHD size is set
            if self.latin_hypercube_n is None:
                raise Exception('No LHD information available. Aborting.')

        # Check run directory
        if self.run_directory is None:
            print('Run directory is unset, assuming pwd.')
            self.run_directory = '.'

    def get_batch_size(self,batch_number):
        """
        Just returns the size of the batch.
        """
        
        if batch_number == 0:
            return self.latin_hypercube_n
        else:
            return self.n_batch

    def run_batch(self, batch_number, max_containers=124 ):
        """
        Runs the current batch of jobs.
        """

        self.check_settings(batch_number)

        # Run directory for this batch
        run_directory = self.run_directory + os.sep + 'batch_' + str(batch_number)

        # Size of this batch
        nruns = self.get_batch_size(batch_number)

        # Run this batch
        super().run(run_directory,nruns,self.image_name,max_containers)

    def recover_batch_output(self,batch_number,wait=True):
        """
        Recovers output after runs have completed.
        """

        self.check_settings(batch_number)

        # Run directory for this batch
        run_directory = self.run_directory + os.sep + 'batch_' + str(batch_number)

        # Size of this batch
        nruns = self.get_batch_size(batch_number)
        
        super().collate_results(run_directory, nruns, filename=self.file_name, wait=wait)

        # Returns input names, current inputs, output names and current outputs.
        return super().get_parameters_and_targets(self.current_pyro_objects)

    def train_initial_design(self,targets):
        """
        Trains MICE design based on the target values for the inital
        hypercube design.
        """

        self.mice_design.set_initial_targets(targets)

    def train_mice_batch(self,targets):
        """
        Trains MICE design based on the results of the latest
        batch.
        """

        self.mice_design.set_batch_targets(targets) 

    def save_design(self,filename):
        """
        Saves the current state of the MICE design.
        """

        self.mice_design.save_design(filename)

    def load_design(self, filename, image_name='gs2_local', input_filename='gs2.in', 
                    directory='.', template_file=None):
        """
        Loads a design from a file.
        """

        self.image_name    = image_name
        self.file_name     = input_filename
        self.run_directory = directory
        self.template_file = template_file

        # Generate a Latin Hypercube
        self.lhd = mogp.LatinHypercubeDesign( len(self.param_dict) )

        # Create a MICE design
        self.mice_design = mogp.MICEDesign( lhd )

        # Load existing data
        self.mice_design.load_design(filename)

    def submit_inital_design(self,image_name='gs2_local',max_containers=124,n_init=124,n_cand=50,
                             file_name='gs2.in',directory='.',template_file=None):
        """
        Sets up and runs the initial LHD design process.
        """

        # Create design
        self.create_design(n_init=n_init, n_cand=n_cand, file_name=file_name, 
                           directory=directory, template_file=template_file)

        # Run initial batch
        self.run_batch( 0, image_name=image_name, max_containers=max_containers )

        # Recover initial batch 
        input_names, inputs, output_names, outputs = self.recover_output(0)

        # Train initial design
        self.train_initial_design(outputs)

        # Write an initial CSV file
        filename = 'batch_0.csv'
        self.create_csv(self.current_pyro_objects,directory,filename)

        # Write an initial MICE file
        filename = 'batch_0.npz'
        self.save_design(filename)

    def submit_mice_batch(batch_number, image_name='gs2_local', n_batch=8, max_containers=124):
        """
        Submits a batch of runs generated using MICE.
        """

        # Set up files for batch of runs
        self.create_mice_batch( batch_number, n_batch=n_batch)

        # Run batch
        self.run_batch( batch_number, image_name=image_name, max_containers=max_containers )

        # Recover batch 
        input_names, inputs, output_names, outputs = self.recover_output(batch_number)

        # Update MICE Design
        self.train_mice_batch(outputs)

        # Write a new CSV file
        filename = 'batch_'+str(batch_number)+'.csv'
        self.create_csv(self.current_pyro_objects,directory,filename)

        # Write an new MICE file
        filename = 'batch_'+str(batch_number)+'.npz'
        self.save_design(filename)
