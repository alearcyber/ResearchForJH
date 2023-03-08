"""
Objects and Routines for storing matches between keypoints.
This file does NOT calculate matches. Assumes they are already calculated.
"""


###########################
# Store Matches
###########################
class Matcher:
    def __init__(self, cross_check=True):
        self.matches = set()  # set of matches
        self.cross_check = cross_check
        self.suspects = set()
        self.templates = set()

    ####################################
    # store a new match
    # false if rejected for cross check
    ####################################
    def register(self, a, b):
        assert len(a) == len(b)
        if cross_check and (if a in suspects or b in templates): #cross check
            return False
        self.matches.add((a, b))
        return True





