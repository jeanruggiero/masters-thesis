"""Script to assign labels to all scans in the s3 bucket."""
from labeling import S3ScanLabeler, BScanMergeCrawler

import pandas as pd

geometry_spec = pd.concat(
    [pd.read_csv(filename, index_col=0) for filename in [
        'geometry_spec.csv', 'geometry_spec2.csv', 'geometry_spec3.csv'
    ]]
)


labeler = S3ScanLabeler('jean-masters-thesis', 'simulations/', geometry_spec)
labels = labeler.label_inside_outside()

print(labels[1023])
#
# crawler = BScanMergeCrawler('jean-masters-thesis', 'simulations/')
# crawler.merge_all()

