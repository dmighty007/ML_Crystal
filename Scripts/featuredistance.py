import numpy as np
import pandas as pd
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.distances import distance_array


class featuredistance(AnalysisBase):  # subclass AnalysisBase

    def __init__(self, atomgroup, contacts, verbose=True):
        """
        Set up the initial analysis parameters.
        """
        # must first run AnalysisBase.__init__ and pass the trajectory

        trajectory = atomgroup.universe.trajectory
        super(featuredistance, self).__init__(trajectory,
                                               verbose=verbose)
        # set atomgroup as a property for access in other methods

        self.atomgroup = atomgroup

        unique_atoms = np.unique(contacts.ravel())

        self.contact_dict = {}
        for count, (i, j) in enumerate(contacts):
           self.contact_dict[count] = [np.where(unique_atoms == i)[0][0], np.where(unique_atoms == j)[0][0]]

    def _prepare(self):
        """
        Create array of zeroes as a placeholder for results.
        This is run before we begin looping over the trajectory.
        """
        # This must go here, instead of __init__, because
        # it depends on the number of frames specified in run().
        self.results = np.zeros((self.n_frames, len(self.contact_dict)))
        self.times = np.zeros(self.n_frames)

    def _single_frame(self):
        """
        This function is called for every frame that we choose
        in run().
        """
        # call our earlier function
        all_distances = distance_array(self.atomgroup.positions, self.atomgroup.positions, self.atomgroup.universe.dimensions)
        distances = [all_distances[self.contact_dict[i][0]][[self.contact_dict[i][1]]][0] for i in range(len(self.contact_dict))]


        # save it into self.results
        self.results[self._frame_index] = np.array(distances)
        self.times[self._frame_index] = self._trajectory.time

    def _conclude(self):
        columns = ['Time (ps)']
        for i in range(len(self.contact_dict)):
            columns.append(f'd_{i}')
        series = np.column_stack([self.times, self.results])
        self.df = pd.DataFrame(series, columns=columns)
