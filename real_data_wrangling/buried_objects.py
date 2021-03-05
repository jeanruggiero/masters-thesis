from typing import Sequence


class BuriedObject:

    def __init__(self, x: float, y: float, size: float, shape: str, profiles: Sequence[int]):
        self.x = x
        self.y = y
        self.size = size
        self.shape = shape
        self.profiles = profiles

    def is_in_scan(self, scan_x1: float, scan_x2: float, scan_profile: int) -> bool:
        """
        Returns True if this object is within the bounds of the provided scan or False otherwise.

        :param scan_x1: x location marking the start of the scan
        :param scan_x2: x location marking the end of the scan
        :param scan_profile: profile number along which the scan was performed
        :return: True if this objects is within the bounds of the provided scan or False otherwise
        """

        if scan_profile not in self.profiles:
            return False

        # Swap x1 and x2 if x1 > x2
        scan_x1, scan_x2 = min(scan_x1, scan_x2), max(scan_x1, scan_x2)

        if self.x < scan_x1 or self.x > scan_x2:
            # Object is outside the window of this scan
            return False

        return True

    def x_in_scan(self, scan_x1, scan_x2, scan_profile):
        """
        Returns the x location of the center of the object within the scan, using the start of the scan as x = 0.

        :param scan_x1: x location marking the start of the scan
        :param scan_x2: x location marking the end of the scan
        :param scan_profile: profile number along which the scan was performed
        :return: the x location of the center of the object within the scan
        :raises ValueError: if this object is not within the provided scan
        """

        if not self.is_in_scan(scan_x1, scan_x2, scan_profile):
            raise ValueError("This object is not in the provided scan.")

        # Swap x1 and x2 if x1 > x2
        return self.x - min(scan_x1, scan_x2)

