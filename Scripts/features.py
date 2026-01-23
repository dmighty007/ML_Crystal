import os
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from featurecoordinate import featurecoordinates

# from featuredistance import featuredistance
from joblib import Parallel, delayed
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import calc_bonds


class featuredistance(AnalysisBase):
    def __init__(self, atomgroup, contacts, verbose=True):
        """
        Optimized Distance Calculation.
        """
        trajectory = atomgroup.universe.trajectory
        super(featuredistance, self).__init__(trajectory, verbose=verbose)
        self.atomgroup = atomgroup

        # --- Vectorized Index Mapping ---
        # We need to map global atom indices (from 'contacts') to local indices
        # (0 to N) relative to the 'atomgroup'.

        # 1. Get the actual global indices of the selected atoms
        #    (Assuming unique and sorted for searchsorted)
        global_indices = atomgroup.indices

        # 2. Prepare contacts array
        contacts_arr = np.array(contacts, dtype=int)

        # 3. Use searchsorted to find where the contact indices fit into the atomgroup
        #    (This replaces the slow dictionary loop)
        sorter = np.argsort(global_indices)
        insert_positions = np.searchsorted(global_indices, contacts_arr, sorter=sorter)

        # 4. Map to local indices
        #    We use the sorter to retrieve the original index in the atomgroup array
        self.local_pairs = sorter[insert_positions]

        # Split into two arrays for fast indexing in _single_frame
        self.idx_A = self.local_pairs[:, 0]
        self.idx_B = self.local_pairs[:, 1]

    def _prepare(self):
        # Pre-allocate results array
        self.results = np.zeros((self.n_frames, len(self.idx_A)), dtype=np.float32)
        self.times = np.zeros(self.n_frames, dtype=np.float32)

    def _single_frame(self):
        # --- Fast Calculation ---
        # Instead of calculating the full N*N matrix, we only calculate
        # the specific pairs we need using calc_bonds (O(K) complexity).

        box = self._trajectory.ts.dimensions
        coords = self.atomgroup.positions

        # Slice coordinates directly
        pos_A = coords[self.idx_A]
        pos_B = coords[self.idx_B]

        # Calculate distances with PBC automatically handled
        self.results[self._frame_index] = calc_bonds(pos_A, pos_B, box=box)
        self.times[self._frame_index] = self._trajectory.time

    def _conclude(self):
        # Optional: Add your DataFrame conversion here if needed
        pass

class Feature:
    def __init__(self, scaler_file=None, model_file=None):
        self.scaler_file = scaler_file
        self.model_file = model_file
        self.contacts = None
        self.atom_index = None
        self.selection = None
        self.mode = None
        self.top = None
        self.traj = None
        self.features = None

    def prepare_contacts(self, contacts=None, contacts_file=None, mode="distance"):
        self.mode = mode
        data = None

        if contacts is not None:
            data = np.array(contacts, dtype=int)
        elif contacts_file is not None:
            data = np.loadtxt(contacts_file, dtype=int)

        if self.mode == "distance":
            if data is not None: self.contacts = data
            unique_atoms = np.unique(self.contacts)
        elif self.mode == "coordinate":
            if data is not None: self.atom_index = data
            self.atoms = self.atom_index
            unique_atoms = np.unique(self.atoms)

        # Ensure correct selection string
        self.selection = "index " + " or index ".join(unique_atoms.astype(str))

    @staticmethod
    def _process_chunk(top_file, traj_file, selection, data_indices, mode, start, stop, step):
        """Thread-safe worker."""
        u = mda.Universe(top_file, traj_file)
        atomgroup = u.select_atoms(selection)

        if mode == "distance":
            # Using the NEW optimized class
            analyzer = featuredistance(atomgroup, contacts=data_indices, verbose=False)
        elif mode == "coordinate":
            analyzer = featurecoordinates(atomgroup)

        analyzer.run(start=start, stop=stop, step=step, verbose=False)
        return analyzer.results

    def feature_extraction(self, top=None, traj=None, step=1, frames=None, verbose=True, mda_universe=None):
        self.frames = frames

        if mda_universe is not None:
            # --- SERIAL MODE (Optimized) ---
            if verbose: print("Running in Memory/Serial mode (Optimized)...")
            universe = mda_universe
            atomgroup = universe.select_atoms(self.selection)

            if self.mode == "distance":
                # The optimized featuredistance class is now much faster here
                fd = featuredistance(atomgroup, self.contacts, verbose=verbose)
                fd.run(step=step, verbose=verbose)
                self.features = fd.results
            elif self.mode == "coordinate":
                fc = featurecoordinates(atomgroup)
                fc.run(step=step, verbose=verbose)
                self.features = fc.results

        else:
            # --- PARALLEL MODE (File-based) ---
            self.top = top
            self.traj = traj
            # Create dummy universe just for frame count
            u_temp = mda.Universe(top, traj) if isinstance(traj, str) else mda.Universe(top, traj[0])
            total_frames = u_temp.trajectory.n_frames

            n_jobs = -1
            cpu_count = os.cpu_count()
            chunk_size = total_frames // cpu_count

            tasks = []
            for i in range(cpu_count):
                start = i * chunk_size
                stop = (i + 1) * chunk_size if i != cpu_count - 1 else total_frames

                tasks.append(delayed(self._process_chunk)(
                    self.top, self.traj, self.selection,
                    self.contacts if self.mode == "distance" else self.atom_index,
                    self.mode, start, stop, step
                ))

            if verbose: print(f"Running parallel feature extraction on {cpu_count} CPUs...")
            results = Parallel(n_jobs=n_jobs)(tasks)
            self.features = np.concatenate(results)

        # Filtering
        if self.frames is not None and len(self.frames) > 0:
            self.features = self.features[self.frames]

        return self.features
