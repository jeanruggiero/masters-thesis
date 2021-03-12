from gprutil.plotters import plot_bscan, subplot_bscans


if __name__ == '__main__':
    output_path = 'simulations/test/'
    # filename = 'test_cylinder_4'
    #
    # plot_bscan(output_path + filename + '_merged.out')

    subplot_bscans(output_path + 'test_cylinder', 10)
