import os
from multiprocessing import cpu_count
from .config import DENDRITES_ONE, NUCLEATION_ONE, BSE_ONE, BSE_TWO, CELL_ONE, CELL_TWO
from .evaluation import calculate_psnr, calculate_ssim


def evaluate_frame(ground_truth, reconstructed):
    psnr = calculate_psnr(ground_truth, reconstructed)
    ssim = calculate_ssim(reconstructed, ground_truth)
    return psnr, ssim


def get_max_parallel_processes():
    try:
        n_cpu = cpu_count()
    except NotImplementedError:
        n_cpu = 1
    # Optionally, add memory checks here
    return max(1, n_cpu)


class ExperimentsSetup:

    SUPPORTED_SAMPLING_METHODS = ['stratified', 'uniform']
    SUPPORTED_INTERPOL_METHODS = ['linear', 'nearest', 'cubic']
    SUPPORTED_GROUND_TRUTHS = {"dendrites_one": DENDRITES_ONE, "nucleation_one": NUCLEATION_ONE,
                               "bse_one": BSE_ONE, "bse_two": BSE_TWO,"cell_one":CELL_ONE,
                               "cell_two": CELL_TWO}

    def __init__(self, numberOfFrames,
                 initialSampling='stratified', interpolMethod='linear', groundTruthName ="dendrites_one"):

        self.groundTruthName = groundTruthName
        self.numberOfFrames = numberOfFrames
        self.imageShape = self.SUPPORTED_GROUND_TRUTHS[groundTruthName][0].shape

        self.initialSampling = initialSampling
        self.interpolMethod = interpolMethod

        self.plots_dir = 'plots'
        os.makedirs(self.plots_dir, exist_ok=True)

    def validate_inputs(self):

        if self.initialSampling not in self.SUPPORTED_SAMPLING_METHODS:
            raise ValueError(f"initialSampling must be one of {self.SUPPORTED_SAMPLING_METHODS}.")

        if self.interpolMethod not in self.SUPPORTED_INTERPOL_METHODS:
            raise ValueError(f"interpolMethod must be one of {self.SUPPORTED_INTERPOL_METHODS}.")

        if self.groundTruthName not in self.SUPPORTED_GROUND_TRUTHS:
            raise ValueError(f"ground truth name must be one of {self.SUPPORTED_GROUND_TRUTHS}.")