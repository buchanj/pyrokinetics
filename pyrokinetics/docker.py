# A file containing tools for submitting runs
# of GS2 using docker.

import os
import multiprocessing
import time

def count_active_containers(image_name):
    """
    Counts the number of currently running containers of the given image
    """

    command = 'docker ps | grep ' + image_name + ' | wc -l'
    count = int( os.popen(command).read() )

    return count

def check_finished(image_name):
    """
    Checks that all runs have finished.
    """

    active = count_active_containers(image_name)
    if active > 0:
        return False
    else:
        return True

def wait_until_finished(image_name,wait_time=60):
    """
    Just waits until all instances of a container
    have stopped.
    """

    while True:

        finished = check_finished(image_name)
        if not finished:
            print('Jobs are still running!')
            time.sleep(wait_time)
        else:
            return

def get_absolute_path(run_directory):
    """
    Docker won't attach volumes using relative paths so need to convert relative
    paths to absolute ones before submitting containers.
    """

    return os.path.abspath(run_directory)

def submit_container(image_name,run_directory,file_name,cores_per_run):
    """
    Submits a container of the given image name in the specified run directory in detached
    mode. Assumes the container is set up to awaken in /tmp/work_dir as for the VVeb.UQ app
    """

    # Pass cores_per_run and input filename as environments variable to docker

    abs_run_directory = get_absolute_path(run_directory)

    command = 'docker run -v ' + abs_run_directory + ':/tmp/work_dir -d --env GS2_CPUS=' + str(cores_per_run) + \
              ' --env INPUT_FILE=' + str(file_name) + ' ' + image_name
    
    print('Submitting container in directory ' + abs_run_directory)
    os.system(command)
    print('Submitted')

def can_run_container(image_name,cores_per_run, max_containers):
    """
    Checks if there is sufficient available cores to run
    another containter based on cores per job.
    """

    active_containers = count_active_containers(image_name)
    print( str(active_containers) + ' active containers' )

    # Check preset maximum
    if( active_containers >=max_containers ):
        return False

    # Get total number of available processors
    total_cores = multiprocessing.cpu_count()

    # Check if there are processors available for new container
    used_cores      = active_containers * cores_per_run
    available_cores = total_cores - used_cores

    if( available_cores >= cores_per_run):
        return True
    else:
        return False

def run_docker_local(directory,file_name,nruns,cores_per_run,image_name,max_containers):
    """ 
    Submits a set of containerised GS2 runs in a folder.
    Currently assumes each run is a single core run.

    directory : Top level directory name to generate iteration folders in. (iteration_#)
    file_name : Name of the gs2 input file
    nruns     : Number of runs to generate
    cores_per_run : Number of cores used by each run (used to determine whether another container can run)
    image_name    : Name of docker image to run in each folder
    max_containers: Maximum set number of containers to use
    """

    # Iterate through all runs and submit output
    for run in range(nruns):

        # Directory name for this particular run
        run_directory = directory + os.sep + 'iteration_'+str(run) + os.sep

        while( True ):

            try:

                # Check if there are processors available for new container
                can_run = can_run_container(image_name,cores_per_run, max_containers)

                if( can_run ):

                    submit_container(image_name,run_directory,file_name,cores_per_run)
                    break

                else:

                    print('Waiting for available cores...')
                    time.sleep(10)

            except:

                print('Error submitting container in directory '+run_directory)
                break
