# Copyright (C) 2015-2020: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from gprMax.exceptions import CmdInputError


def get_output_data(filename, rxnumber, rxcomponent):
    """Gets B-scan output data from a model.

    Args:
        filename (string): Filename (including path) of output file.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
    """

    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(filename))

    path = '/rxs/rx' + str(rxnumber) + '/'
    availableoutputs = list(f[path].keys())

    # Check if requested output is in file
    if rxcomponent not in availableoutputs:
        raise CmdInputError('{} output requested to plot, but the available output for receiver 1 is {}'.format(rxcomponent, ', '.join(availableoutputs)))

    outputdata = f[path + '/' + rxcomponent]
    outputdata = np.array(outputdata)
    f.close()

    return outputdata, dt


def mpl_plot(filename, outputdata, dt, rxnumber, rxcomponent):
    """Creates a plot (with matplotlib) of the B-scan.

    Args:
        filename (string): Filename (including path) of output file.
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        plt (object): matplotlib plot object.
    """

    (path, filename) = os.path.split(filename)

    fig = plt.figure(num=filename + ' - rx' + str(rxnumber),
                     figsize=(20, 10), facecolor='w', edgecolor='w')
    plt.imshow(outputdata,
               extent=[0, outputdata.shape[1], outputdata.shape[0] * dt, 0],
               interpolation='nearest', aspect='auto', cmap='seismic',
               vmin=-np.amax(np.abs(outputdata)), vmax=np.amax(np.abs(outputdata)))
    plt.xlabel('Trace number')
    plt.ylabel('Time [s]')
    # plt.title('{}'.format(filename))

    # Grid properties
    ax = fig.gca()
    ax.grid(which='both', axis='both', linestyle='-.')

    cb = plt.colorbar()
    if 'E' in rxcomponent:
        cb.set_label('Field strength [V/m]')
    elif 'H' in rxcomponent:
        cb.set_label('Field strength [A/m]')
    elif 'I' in rxcomponent:
        cb.set_label('Current [A]')

    # Save a PDF/PNG of the figure
    # savefile = os.path.splitext(filename)[0]
    # fig.savefig(path + os.sep + savefile + '.pdf', dpi=None, format='pdf',
    #             bbox_inches='tight', pad_inches=0.1)
    # fig.savefig(path + os.sep + savefile + '.png', dpi=150, format='png',
    #             bbox_inches='tight', pad_inches=0.1)

    return plt


def plot_bscan(filename: str, rx_component: str = 'Ez'):
    # Open output file and read number of outputs (receivers)
    with h5py.File(filename, 'r') as f:
        nrx = f.attrs['nrx']

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError(f'No receivers found in {filename}')

    for rx in range(1, nrx + 1):
        outputdata, dt = get_output_data(filename, rx, rx_component)
        plthandle = mpl_plot(filename, outputdata, dt, rx, rx_component)

    plthandle.show()