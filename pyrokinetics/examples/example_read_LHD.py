from pyrokinetics import Pyro, PyroScan, PyroLHD
import os
import numpy as np

# Point to input files
templates = os.path.join('..', 'templates')

# Equilibrium file
eq_file = os.path.join(templates, 'test.geqdsk')

# Kinetics data file
kinetics_file = os.path.join(templates, 'jetto.cdf')

# Load up pyro object
pyro = Pyro(eq_file=eq_file, eq_type='GEQDSK', kinetics_file=kinetics_file, kinetics_type='JETTO')

# Generate local parameters at psi_n=0.5
pyro.load_local(psi_n=0.5, local_geometry='Miller')

# Change GK code to GS2
pyro.gk_code = 'GS2'

# Write single input file using my own template
pyro.write_gk_file(file_name='test_jetto.gs2', template_file='step.in')

# Use existing parameter
param_1 = 'ky'
values_1 = np.arange(0.5, 0.6, 0.1)

# Dictionary of param and values
param_dict = {param_1: values_1}

# Create PyroScan object
pyro_scan = PyroLHD(pyro,
                    param_dict,
                    value_fmt='.3f',
                    value_separator='_',
                    parameter_separator='_',
                    file_name='step.in'
)

#pyro_scan.write(npoints=3, directory='test_output')

image_name     = 'gs2_local'
max_containers = 10 

#pyro_scan.run_docker_local(image_name,max_containers)

pyro_scan.latin_hypercube_n = 3
pyro_scan.run_directory = './test_output/'

print('Collating LHD results')
pyro_scan.collate_results()

print('Creating output csv')
pyro_scan.create_csv()
