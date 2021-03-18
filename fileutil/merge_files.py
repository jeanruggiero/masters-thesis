import glob
import h5py
import os
import io
import re
import boto3


def merge_files(basefilename, s3_bucket=None, removefiles=False):
    """Merges traces (A-scans) from multiple output files into one new file,
        then optionally removes the series of output files.

    Args:
        basefilename (string): Base name of output file series including path.
        outputs (boolean): Flag to remove individual output files after merge.
    """

    # Open output file for writing
    with h5py.File('merged.out', 'w') as fout:

        if s3_bucket:
            merge_s3_objects(basefilename, s3_bucket, fout, removefiles)

        else:
            merge_local_files(basefilename, fout, removefiles)


def merge_local_files(basefilename, fout, removefiles):
    # Read files from local machine
    files = glob.glob(basefilename + '*.out')
    output_files = [filename for filename in files if '_merged' not in filename]
    n_runs = len(output_files)

    # Add positional data for rxs
    for model in range(n_runs):

        with h5py.File(basefilename + str(model + 1) + '.out', 'r') as f:
            nrx = f.attrs['nrx']

            # Write properties for merged file on first iteration
            if model == 0:
                fout.attrs['Title'] = f.attrs['Title']
                fout.attrs['gprMax'] = '3.1.5'
                fout.attrs['Iterations'] = f.attrs['Iterations']
                fout.attrs['dt'] = f.attrs['dt']
                fout.attrs['nrx'] = f.attrs['nrx']
                for rx in range(1, nrx + 1):
                    path = '/rxs/rx' + str(rx)
                    grp = fout.create_group(path)
                    availableoutputs = list(f[path].keys())
                    for output in availableoutputs:
                        grp.create_dataset(output, (fout.attrs['Iterations'], n_runs),
                                           dtype=f[path + '/' + output].dtype)

            # For all receivers
            for rx in range(1, nrx + 1):
                path = '/rxs/rx' + str(rx) + '/'
                availableoutputs = list(f[path].keys())
                # For all receiver outputs
                for output in availableoutputs:
                    fout[path + '/' + output][:, model] = f[path + '/' + output][:]

    if removefiles:
        for model in range(n_runs):
            file = basefilename + str(model + 1) + '.out'
            os.remove(file)


def merge_s3_objects(basefilename, bucket, fout, removefiles):
    # Read files from s3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name="jean-masters-thesis")

    # Get all objects matching basefilename pattern and then sort by ascan number
    objs = sorted(list(bucket.objects.filter(Prefix=basefilename + '/')),
        key=(lambda obj: int(re.findall(r'\d+', obj.key.split('/')[-1])[0]))
    )
    n_runs = len(objs)

    for i, obj in enumerate(objs):

        with io.BytesIO() as b:
            s3.Object(bucket.name, obj.key).download_fileobj(b)

            with h5py.File(b, 'r') as f:

                nrx = f.attrs['nrx']

                # Write properties for merged file on first iteration
                if i == 0:
                    fout.attrs['Title'] = f.attrs['Title']
                    fout.attrs['gprMax'] = '3.1.5'
                    fout.attrs['Iterations'] = f.attrs['Iterations']
                    fout.attrs['dt'] = f.attrs['dt']
                    fout.attrs['nrx'] = f.attrs['nrx']
                    for rx in range(1, nrx + 1):
                        path = '/rxs/rx' + str(rx)
                        grp = fout.create_group(path)
                        availableoutputs = list(f[path].keys())
                        for output in availableoutputs:
                            grp.create_dataset(output, (fout.attrs['Iterations'], n_runs),
                                               dtype=f[path + '/' + output].dtype)

                # For all receivers
                for rx in range(1, nrx + 1):
                    path = '/rxs/rx' + str(rx) + '/'
                    availableoutputs = list(f[path].keys())
                    # For all receiver outputs
                    for output in availableoutputs:
                        fout[path + '/' + output][:, i] = f[path + '/' + output][:]

    if removefiles:
        for obj in objs:
            s3.Object(bucket.name, obj.key).delete()