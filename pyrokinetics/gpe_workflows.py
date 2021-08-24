from pyrokinetics import Pyro, PyroScan, PyroScan_LHD, PyroScan_MICE, PyroGPE
import os
import numpy as np
import argparse

def run_lhd_workflow(gs2_template, param_dict, npoints, directory, image_name='gs2_local',
                     max_containers=124, wait=True, test_mode=False):
    """
    Generates, submit an LHD workflow.
    Also recovers and saves an LHD object if wait is true.
    
    gs2_template   : Template input file to base LHD on
    param_dict     : dictionary of parameter ranges to sample 
                     Each entry must be param_name : np.array([min_val, max_val])
    npoints        : Number of points to sample
    max_containers : Maximum number of containers to run simultaneously.
    test_mode      : Sets test mode (No containers)

    """

    # Load up pyro object
    pyro = Pyro(gk_file=gs2_template, gk_type='GS2')

    # Create PyroScan object
    pyro_scan = PyroScan_LHD(pyro,
                             directory = directory,
                             image_name = image_name,
                             template_file = gs2_template,
                             param_dict = param_dict,
                             value_fmt='.3f',
                             value_separator='_',
                             parameter_separator='_',
                             test_mode = test_mode
                         )

    pyro_scan.submit(npoints=npoints, max_containers=max_containers, wait=wait)

    return pyro_scan

def run_lhd_gpe_workflow(gs2_template, param_dict, npoints, directory, image_name='gs2_local',
                         max_containers=124, test_mode=False):
    """
    Generates and submit an LHD workflow. Then trains a GPE on the result and performs
    Leave One Out Cross Validation.
    
    gs2_template   : Template input file to base LHD on
    param_dict     : dictionary of parameter ranges to sample 
                     Each entry must be param_name : np.array([min_val, max_val])
    npoints        : Number of points to sample
    max_containers : Maximum number of containers to run simultaneously.
    test_mode      : Sets test mode (No containers)

    """

    pyro_scan = run_lhd_workflow(gs2_template, param_dict, npoints, directory, image_name=image_name,
                                 max_containers=max_containers, test_mode=test_mode)

    # Train a GPE using the stored data
    pyro_gpe = PyroGPE()

    # Load training data from LHD object
    print('Reading training data from LHD object')
    pyro_gpe.load_training_data_from_lhd(pyro_scan)

    # Train GPEs based on loaded data
    print('Training GPEs')
    pyro_gpe.train()

    # Perform leave one out cross validation
    frequency_rms_error, gamma_rms_error, frequency_zs, gamma_zs = pyro_gpe.leave_one_out_cross_validate()
    print()
    print( 'Frequency RMS error / range : '+str(frequency_rms_error) )
    print()
    print( 'Growth Rate RMS error / range : '+str(gamma_rms_error) )

    # Write GPEs to file so they can be read in later
    print('Saving GPEs to file')
    pyro_gpe.save_gpes()

    return pyro_gpe

# FIXME - Add a convergence termination criterion
# FIXME - What to do about theta values.    

def run_mice_workflow(gs2_template, param_dict, directory, image_name='gs2_local',
                      max_containers=124, n_init=124, n_cand=50, n_batch=20, 
                      n_iterations=10, validation_lhd=None, test_mode=False):

    """
    Submits a MICE workflow
    
    gs2_template   : Template input file to base LHD on
    param_dict     : dictionary of parameter ranges to sample 
                     Each entry must be param_name : np.array([min_val, max_val])
    directory      : Top level directory name to store data
    image_name     : Name of docker image to run
    max_containers : Maximum number of containers to run simultaneously.
    n_init         : Size of initial LHD 
    n_batch        : Size of each batch of MICE runs
    n_cand         : Number of candidate points to use when running MICE
    n_iterations   : Number of MICE iterations to make
    validate       : An LHD to cross validate against

    """

    # Load up pyro object
    pyro = Pyro(gk_file=gs2_template, gk_type='GS2')

    # Create PyroScan object
    pyro_scan = PyroScan_MICE(pyro,
                              directory = directory,
                              image_name = image_name,
                              template_file = gs2_template,
                              param_dict = param_dict,
                              value_fmt='.3f',
                              value_separator='_',
                              parameter_separator='_',
                              test_mode = test_mode
                          )

    # Generate and run initial LHD 
    # This also updates the MICE design with the resulting targets
    pyro_scan.submit_inital_design(max_containers=max_containers, n_init=n_init, n_cand=n_cand)

    # Train initial GPE based on MICE inputs and targets
    pyro_gpe = PyroGPE()
    pyro_gpe.load_training_data_from_mice(pyro_scan)
    pyro_gpe.train(kernel='Matern52',nugget=1.0e-8)

    # Perform validation using initial LHD
    freq_rms_error, gamma_rms_error, freq_zs, gamma_zs = pyro_gpe.leave_one_out_cross_validate()

    filename = directory + os.step + 'loocv.csv'
    pyro_gpe.write_validation_data( 0, filename, freq_rms_error, gamma_rms_error )

    # Perform validation against LHD
    if validation_lhd is not None:

        freq_rms_error, gamma_rms_error, freq_zs, gamma_zs = pyro_gpe.validate_from_lhd(validation_lhd)

        filename = directory + os.step + 'validation.csv'
        pyro_gpe.write_validation_data( 0, filename, freq_rms_error, gamma_rms_error )

    iteration = 1
    while iteration <= n_iterations:

        # Submit a batch 
        pyro_scan.submit_mice_batch(iteration, n_batch=n_batch, max_containers=124)

        # Train a new GPE using updated data
        pyro_gpe.load_training_data_from_mice(pyro_scan)
        pyro_gpe.train(kernel='Matern52',nugget=1.0e-8)

        # Perform validation using initial LHD
        freq_rms_error, gamma_rms_error, freq_zs, gamma_zs = pyro_gpe.leave_one_out_cross_validate()

        filename = directory + os.step + 'loocv.csv'
        pyro_gpe.write_validation_data( iteration, filename, freq_rms_error, gamma_rms_error )

        # Perform validation against LHD
        if validation_lhd is not None:

            freq_rms_error, gamma_rms_error, freq_zs, gamma_zs = pyro_gpe.validate_from_lhd(validation_lhd)

            filename = directory + os.step + 'validation.csv'
            pyro_gpe.write_validation_data( iteration, filename, freq_rms_error, gamma_rms_error )

    return pyro_scan, pyro_gpe

def load_param_dict(filename='params.in'):
    """
    Loads a file containing parameter data.
    Expected format is a set of lines for the form:
    param name min_val max_val
    """

    param_dict = {}
    with open(filename) as fhandle:

        for line in fhandle:

            words = line.strip().split()
            param_dict[ words[0] ] = [ float(words[1]) , float(words[2]) ]

    return param_dict
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Specify the parameter input file
    helpstr = "Name of the file containing the parameter data to read."
    parser.add_argument("-p", "--parameters", default='params.in', help=helpstr)

    # Specify the number of cores to use 
    helpstr = "Number of cores to use per run for an LHD workflow (incl. start of MICE design)."
    parser.add_argument("-nl", "--ncores_lhd", default=1, type=int, help=helpstr)

    # Number of cores to use per MICE batch
    helpstr = "Number of cores to use per run for a MICE batch"
    parser.add_argument("-nm", "--ncores_mice", default=1, type=int, help=helpstr)

    # Specify the GS2 template namelist file to use.  
    helpstr = "Specify an template input namelist to use"
    parser.add_argument("-t", "--template", default='gs2.in', help=helpstr)

    # Specify the number of points to use for an LHD
    helpstr = "Specify the number of points to use for an LHD."
    parser.add_argument("-np",   "--npoints", default=124, type=int, help=helpstr)

    # Specify the top level directory for the workflow
    helpstr = "Sets the top level directory."
    parser.add_argument("-d", "--dir", default='./', help=helpstr)

    # Specify the docker image for the run
    helpstr = "Sets the docker image name"
    parser.add_argument("-i", "--image", default='gs2_local', help=helpstr)

    # Specify a maximum number of containers to use
    helpstr = "Sets the maximum number of containers"
    parser.add_argument("-c", "--max_containers", default=124, help=helpstr)

    # Wait for completion
    helpstr = "Waits for an LHD run to complete and aggregates data."
    parser.add_argument("-w", "--wait", default=True, help=helpstr)

    # Train a GPE on the LHD and cross validate
    helpstr = "Set True to train a GPE on the LHD and perform Leave One Out Cross Validation."
    parser.add_argument("-g", "--gpe", type=bool, default=False, help=helpstr)

    # Run MICE workflow
    helpstr = "Set True to run a full MICE workflow rather than an LHD."
    parser.add_argument("-m", "--mice", type=bool, default=False, help=helpstr)

    # Number of candidate points to use when running MICE
    helpstr = "Sets the number of candidate points to use for a MICE workflow."
    parser.add_argument("-nc", "--n_cand", default=50, help=helpstr)

    # Number of batch points per MICE iteration
    helpstr = "Sets the number of batch points to use for a MICE workflow."
    parser.add_argument("-nb", "--n_batch", default=20, help=helpstr)

    # Number of MICE iterations to perform
    helpstr = "Sets the number of iterations to use for a MICE workflow."
    parser.add_argument("-it", "--iterations", default=10, help=helpstr)
    
    # Test mode
    helpstr = "Run a test - doesn't run containers, just returns 0,1 for freq and gamma."
    parser.add_argument("-tm", "--test_mode", type=bool, default=False, help=helpstr)

    args = parser.parse_args()

    # Load parameter dictionary
    param_dict = load_param_dict(filename=args.parameters)

    if args.mice:

        # FIXME - Handle validation lhd

        run_mice_workflow(args.template, param_dict, args.dir, image_name=args.image,
                          max_containers=args.max_containers, n_init=args.npoints, 
                          n_cand=args.n_cand, n_batch=args.n_batch, n_iterations=args.iterations, 
                          validation_lhd=None,test_mode=args.test_mode)
            
    else:

        if args.gpe:

            run_lhd_gpe_workflow(args.template, param_dict, args.npoints, args.dir, image_name=args.image,
                                 max_containers=args.max_containers, test_mode=args.test_mode)

        else:
            
            run_lhd_workflow(args.template, param_dict, args.npoints, args.dir, image_name=args.image,
                             max_containers=args.max_containers, wait=args.wait, test_mode=args.test_mode)


