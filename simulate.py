from gprutil.plotters import plot_bscan

import os
import glob
import subprocess

import h5py

def merge_files(basefilename, removefiles=False):
    """Merges traces (A-scans) from multiple output files into one new file,
        then optionally removes the series of output files.

    Args:
        basefilename (string): Base name of output file series including path.
        outputs (boolean): Flag to remove individual output files after merge.
    """

    outputfile = basefilename + '_merged.out'
    files = glob.glob(basefilename + '*.out')
    outputfiles = [filename for filename in files if '_merged' not in filename]
    modelruns = len(outputfiles)

    # Combined output file
    fout = h5py.File(outputfile, 'w')

    # Add positional data for rxs
    for model in range(modelruns):
        fin = h5py.File(basefilename + str(model + 1) + '.out', 'r')
        nrx = fin.attrs['nrx']

        # Write properties for merged file on first iteration
        if model == 0:
            fout.attrs['Title'] = fin.attrs['Title']
            fout.attrs['gprMax'] = '3.1.5'
            fout.attrs['Iterations'] = fin.attrs['Iterations']
            fout.attrs['dt'] = fin.attrs['dt']
            fout.attrs['nrx'] = fin.attrs['nrx']
            for rx in range(1, nrx + 1):
                path = '/rxs/rx' + str(rx)
                grp = fout.create_group(path)
                availableoutputs = list(fin[path].keys())
                for output in availableoutputs:
                    grp.create_dataset(output, (fout.attrs['Iterations'], modelruns), dtype=fin[path + '/' + output].dtype)

        # For all receivers
        for rx in range(1, nrx + 1):
            path = '/rxs/rx' + str(rx) + '/'
            availableoutputs = list(fin[path].keys())
            # For all receiver outputs
            for output in availableoutputs:
                fout[path + '/' + output][:, model] = fin[path + '/' + output][:]

        fin.close()

    fout.close()

    if removefiles:
        for model in range(modelruns):
            file = basefilename + str(model + 1) + '.out'
            os.remove(file)


if __name__ == '__main__':
    input_path = 'geometry/test/'
    output_path = 'simulations/test/'
    filename = 'test_cylinder_0'

    input_file = input_path + filename + '.in'

    subprocess.run(['python', '-m', 'gprMax', input_file, '-n 43'])
    merge_files(output_path + filename)
    plot_bscan(output_path + filename + '_merged.out')