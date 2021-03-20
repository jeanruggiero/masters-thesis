from fileutil import merge_files
from gprutil.plotters import plot_bscan

merge_files("simulations/1106", s3_bucket="jean-masters-thesis")
plot_bscan("merged.out")
