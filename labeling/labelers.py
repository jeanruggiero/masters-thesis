import boto3


class S3ScanLabeler:

    def __init__(self, bucket_name, scan_path, geometry_spec):
        s3 = boto3.resource('s3')
        self.bucket = s3.Bucket(name=bucket_name)
        self.scan_path = scan_path
        self.geometry_spec = geometry_spec

    def scans(self):
        """Returns an iterable of scan numbers found in the S3 bucket."""
        keys = [obj.key[len(self.scan_path):] for obj in self.bucket.objects.filter(Prefix=self.scan_path)]
        return set([int(key.split('/')[0]) for key in keys if key.split('/')[0].isnumeric()])

    def label_scan_inside_outside(self, scan_number):
        ascans = 144
        spatial_resolution = 0.002
        scan_start = spatial_resolution * 10 + 0.02
        step_size = 0.02
        scan_locations = (scan_start + i * step_size for i in range(ascans))

        obj_start = self.scan_x(scan_number) - self.scan_radius(scan_number)
        obj_end = self.scan_x(scan_number) + self.scan_radius(scan_number)

        # 0: outside object
        # 1: object start (first ascan taken above object)
        # 2: inside object

        labels = []

        for i, location in enumerate(scan_locations):
            if location < obj_start or location > obj_end:
                labels.append(0)
            else:
                labels.append(1)

        return labels

    def scan_x(self, scan_number):
        return 1.5

    def scan_radius(self, scan_number):
        return self.geometry_spec.loc[scan_number]['radius']

    def label_inside_outside(self):
        return {scan_number: self.label_scan_inside_outside(scan_number) for scan_number in self.scans()}
