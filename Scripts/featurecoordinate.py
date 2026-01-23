import numpy as np
from MDAnalysis.analysis.base import AnalysisBase


class featurecoordinates(AnalysisBase):  # subclass AnalysisBase

    def __init__(self, atomgroup, verbose=True):
        """
        Set up the initial analysis parameters.
        """
        # must first run AnalysisBase.__init__ and pass the trajectory
        trajectory = atomgroup.universe.trajectory
        super(featurecoordinates, self).__init__(trajectory,
                                               verbose=verbose)
        # set atomgroup as a property for access in other methods
        self.atomgroup = atomgroup
        #self.shift_scale = shift_scale
        self.n_atoms = self.atomgroup.n_atoms
        # we can calculate masses now because they do not depend
        # on the trajectory frame.
        #self.masses = self.atomgroup.masses
        #self.total_mass = np.sum(self.masses)

    def _prepare(self):
        """
        Create array of zeroes as a placeholder for results.
        This is run before we begin looping over the trajectory.
        """
        # This must go here, instead of __init__, because
        # it depends on the number of frames specified in run().
        self.results = np.zeros((self.n_frames, self.n_atoms * 3))
        # We put in 6 columns: 1 for the frame index,
        # 1 for the time, 4 for the radii of gyration

    def _single_frame(self):
        """
        This function is called for every frame that we choose
        in run().
        """
        # call our earlier function
        pos = self.atomgroup.positions
        #if self.shift_scale:
        #    pos = pos - pos[0]
        pos = pos.ravel()
        # save it into self.results
        self.results[self._frame_index, :] = pos

    def _conclude(self):
        """
        Finish up by calculating an average and transforming our
        results into a DataFrame.
        """
        return None
