# Tools for reading and writing GPE training data
# to and from csv files.

import csv
import os
import numpy as np

def create_csv(directory, filename, input_names, inputs, output_names, outputs):
    """
    Creates a CSV file containing the varied parameter data and resulting growth rates
    The file is named <filename> in <directory>. It contains a header line with the
    input and output variable names with the inputs followed by the outputs.
    """

    with open( directory + os.sep + filename, 'w' ) as csvfile:

        # CSV writer
        csvwriter = csv.writer(csvfile, delimiter=',')

        # Create a header line containing the varied parameters and growth rates
        headers = input_names + output_names
        csvwriter.writerow(headers)

        # Iterate through all runs and recover output
        for run in range(len(inputs)):

            data = np.array( list(inputs[run]) + list(outputs[run]) )
            csvwriter.writerow(data)

def read_csv(csvfile,n_outputs=2):
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
        target_names    = headers[-1*n_outputs:]

        for row in csvreader:
            parameter_values.append([ float(x) for x in row[:-1*n_outputs] ])
            target_values.append(   [ float(x) for x in row[-1*n_outputs:] ])

    return parameter_names, np.array(parameter_values), target_names, np.array(target_values)
