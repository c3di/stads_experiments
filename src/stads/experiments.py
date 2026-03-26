import os
import numpy as np
import cv2
from datetime import datetime
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from .stads import AdaptiveSampler
from .random_sampler import RandomSampler
from .stratified_sampler import StratifiedSampler
from .interpolator import ImageInterpolator
from .microscope import Microscope
from .experiments_setup import ExperimentsSetup, evaluate_frame, get_max_parallel_processes


def run_adaptive_helper(args):
    initialSampling, interpolMethod, sparsity, numberOfFrames, groundTruthName = args
    sampler = AdaptiveSampler(initialSampling, interpolMethod, sparsity, numberOfFrames, groundTruthName)
    _, psnrs, ssims = sampler.run()
    return sparsity, np.mean(psnrs), np.std(psnrs), np.mean(ssims), np.std(ssims)


def run_static_samplers_helper(args):
    sampler_class, sparsity, interpolMethod, numberOfFrames, groundTruthName = args
    psnrs, ssims = [], []
    microscope = Microscope(groundTruthName)
    imageShape = microscope.imageShape

    if sampler_class.__name__ == "AdaptiveSampler":
        initialSampling = "stratified"
        sampler = sampler_class(initialSampling, interpolMethod, sparsity, numberOfFrames, groundTruthName)
    else:
        sampler = sampler_class(interpolMethod, sparsity, numberOfFrames, groundTruthName)

    for frame_idx in range(numberOfFrames):
        y, x = sampler.get_coordinates()
        pixel_values = microscope.sample_image(y, x, frame_idx)
        intensities = pixel_values[y, x]
        known = np.column_stack((x, y))
        interp = ImageInterpolator(imageShape, known, intensities, interpolMethod)
        image = interp.interpolate_image().clip(0, 255).astype(np.uint8)
        psnr, ssim = evaluate_frame(microscope.groundTruthVideo[frame_idx], image)
        psnrs.append(psnr)
        ssims.append(ssim)

    return sparsity, np.mean(psnrs), np.std(psnrs), np.mean(ssims), np.std(ssims)


def run_adaptive_compare_helper(args):
    sparsity, interpolMethod, numberOfFrames, groundTruthName, initialSampling = args
    sampler = AdaptiveSampler(initialSampling, interpolMethod, sparsity, numberOfFrames, groundTruthName)
    _, psnrs, ssims = sampler.run()
    return sparsity, np.mean(psnrs), np.std(psnrs), np.mean(ssims), np.std(ssims)


class ScalabilityTest(ExperimentsSetup):
    def __init__(self, numberOfFrames, **kwargs):
        super().__init__(numberOfFrames, **kwargs)
        self.sparsityPercents = [10, 15, 20, 30, 50, 75]
        self.validate_inputs()
        self.figures = []

    def validate_inputs(self):
        super().validate_inputs()
        if not isinstance(self.sparsityPercents, list) or not all(isinstance(p, (int, float)) and 0 < p <= 100 for p in self.sparsityPercents):
            raise ValueError("sparsityPercents must be a list of numbers between 0 and 100.")

    def run(self):
        print(f"{datetime.now().strftime('%H:%M:%S')} [ScalabilityTest] Running adaptive samplers in parallel...")
        max_processes = get_max_parallel_processes()

        adaptive_args = [(self.initialSampling, self.interpolMethod, sp, self.numberOfFrames,
                          self.groundTruthName) for sp in self.sparsityPercents]
        with ProcessPoolExecutor(max_workers=min(len(adaptive_args), max_processes)) as executor:
            results_list = list(executor.map(run_adaptive_helper, adaptive_args))
            adaptives = {item[0]: item[1:] for item in results_list}

        results = {}

        for sampler_class, label in zip([RandomSampler, StratifiedSampler], ["uniform", "stratified"]):
            print(f"{datetime.now().strftime('%H:%M:%S')} [ScalabilityTest] Running static sampler '{label}' in parallel...")
            static_args = [(sampler_class, sp, self.interpolMethod, self.numberOfFrames,
                          self.groundTruthName) for sp in self.sparsityPercents]
            with ProcessPoolExecutor(max_workers=min(len(static_args), max_processes)) as executor:
                results_list = list(executor.map(run_static_samplers_helper, static_args))
                sampler_results = {item[0]: item[1:] for item in results_list}
            results[label] = sampler_results

        results['adaptive'] = adaptives

        print(f"{datetime.now().strftime('%H:%M:%S')} [ScalabilityTest] Run complete. Creating plots...")
        fig_psnr, ax_psnr = plt.subplots()
        fig_ssim, ax_ssim = plt.subplots()

        for label, data in results.items():
            x = sorted(data.keys())
            psnr_vals = [data[k][0] for k in x]
            psnr_errs = [data[k][1] for k in x]
            ssim_vals = [data[k][2] for k in x]
            ssim_errs = [data[k][3] for k in x]
            ax_psnr.errorbar(x, psnr_vals, yerr=psnr_errs, label=label, marker='o')
            ax_ssim.errorbar(x, ssim_vals, yerr=ssim_errs, label=label, marker='o')

        ax_psnr.set_title("Mean PSNR vs Sparsity")
        ax_ssim.set_title("Mean SSIM vs Sparsity")
        ax_psnr.set_xlabel("Sparsity (%)")
        ax_ssim.set_xlabel("Sparsity (%)")
        ax_psnr.set_ylabel("PSNR")
        ax_ssim.set_ylabel("SSIM")
        ax_psnr.legend()
        ax_ssim.legend()

        self.figures = [fig_psnr, fig_ssim]
        print(f"{datetime.now().strftime('%H:%M:%S')} [ScalabilityTest] Plots ready.")

    def show_plots(self):
        if not self.figures:
            raise RuntimeError("No plots available. Run the experiment first.")
        print(f"{datetime.now().strftime('%H:%M:%S')} [ScalabilityTest] Showing plots...")
        for fig in self.figures:
            fig.canvas.manager.set_window_title("Scalability Test Plots")
        plt.show()

    def save_plots(self, prefix="scalability"):
        if not self.figures:
            raise RuntimeError("No plots available. Run the experiment first.")
        print(f"{datetime.now().strftime('%H:%M:%S')} [ScalabilityTest] Saving plots...")
        for i, fig in enumerate(self.figures, 1):
            filename = os.path.join(self.plots_dir, f"{prefix}__{self.groundTruthName}_{i}.png")
            fig.savefig(filename)
            print(f"Saved plot to {filename}")

class FlowTest(ExperimentsSetup):

    def __init__(self, numberOfFrames, sparsityPercent=50, **kwargs):
        super().__init__(numberOfFrames, **kwargs)
        self.sparsityPercent = sparsityPercent
        self.figures = []
        self.plots_dir = "plots"
        os.makedirs(self.plots_dir, exist_ok=True)

        self.sampler_no_temporal = None
        self.sampler_with_temporal = None

        self.validate_inputs()

    def validate_inputs(self):
        super().validate_inputs()
        if not isinstance(self.sparsityPercent, (int, float)) or not (0 < self.sparsityPercent <= 100):
            raise ValueError("sparsityPercent must be a number between 0 and 100.")

    def _run_helper(self, withTemporal):

        sampler = AdaptiveSampler(initialSampling=self.initialSampling, interpolMethod=self.interpolMethod,
                                  sparsityPercent=self.sparsityPercent, numberOfFrames=self.numberOfFrames,
                                  groundTruthName=self.groundTruthName, withTemporal=withTemporal)
        reconstructed, psnrs, ssims = sampler.run()
        label = "adaptive_with_temporal" if withTemporal else "adaptive_no_temporal"
        return label, sampler, (list(range(self.numberOfFrames)), psnrs, ssims)


    def run(self, use_parallel=True):

        print(f"{datetime.now().strftime('%H:%M:%S')} [FlowTest] Running flow tests...")
        results = {}

        if use_parallel:
            # Parallel execution of both samplers
            with ProcessPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(self._run_helper, [False, True]))

            # Store results and samplers
            for label, sampler, metrics in outputs:
                results[label] = metrics
                if "with_temporal" in label:
                    self.sampler_with_temporal = sampler
                else:
                    self.sampler_no_temporal = sampler

        else:
            # Sequential fallback
            for temporal in [False, True]:
                label, sampler, metrics = self._run_helper(temporal)
                results[label] = metrics
                if temporal:
                    self.sampler_with_temporal = sampler
                else:
                    self.sampler_no_temporal = sampler

        print(f"{datetime.now().strftime('%H:%M:%S')} [FlowTest] Run complete. Creating plots...")
        fig_psnr, ax_psnr = plt.subplots()
        fig_ssim, ax_ssim = plt.subplots()

        for label, (frames, psnrs, ssims) in results.items():
            ax_psnr.plot(frames, psnrs, label=label, marker="o")
            ax_ssim.plot(frames, ssims, label=label, marker="o")

        ax_psnr.set_title("PSNR per Frame")
        ax_ssim.set_title("SSIM per Frame")
        ax_psnr.set_xlabel("Frame Number")
        ax_ssim.set_xlabel("Frame Number")
        ax_psnr.set_ylabel("PSNR (dB)")
        ax_ssim.set_ylabel("SSIM")
        ax_psnr.legend()
        ax_ssim.legend()

        self.figures = [fig_psnr, fig_ssim]
        print("[FlowTest] Plots ready.")

    def save_plots(self, prefix="flowtest"):
        if not self.figures:
            raise RuntimeError("No plots available. Run the experiment first.")
        print(f"[FlowTest] Saving plots...")
        for i, fig in enumerate(self.figures, 1):
            filename = os.path.join(self.plots_dir, f"{prefix}__{self.groundTruthName}_{i}.png")
            fig.savefig(filename, dpi=200, bbox_inches="tight")
            print(f" Saved plot → {filename}")

    def save_figures(self):
        print(f"[FlowTest] Saving sampling mask and adaptive visuals...")

        if self.sampler_no_temporal is not None:
            print("[FlowTest] Saving 'no temporal' sampler figures...")
            self.sampler_no_temporal.save_figures()
        else:
            print("[FlowTest] No 'no temporal' sampler found — skipping.")

        if self.sampler_with_temporal is not None:
            print("[FlowTest] Saving 'with temporal' sampler figures...")
            self.sampler_with_temporal.save_figures()
        else:
            print("[FlowTest] No 'with temporal' sampler found — skipping.")


class WindowSizeTest(ExperimentsSetup):
    def __init__(self, numberOfFrames=10, sparsityPercent=50, windowSizes=None, **kwargs):
        super().__init__(numberOfFrames, **kwargs)
        self.sparsityPercent = sparsityPercent
        self.windowSizes = windowSizes or [8, 16, 32, 64, 128]
        self.figures = []

    def _run_window_size(self, w):
        sampler = AdaptiveSampler(self.initialSampling, self.interpolMethod, self.sparsityPercent, self.numberOfFrames,
                                  withTemporal=True)
        sampler.windowSize = w
        _, psnrs, ssims = sampler.run()
        return w, np.mean(psnrs), np.mean(ssims)

    def run(self):
        max_processes = get_max_parallel_processes()
        with ProcessPoolExecutor(max_workers=min(len(self.windowSizes), max_processes)) as executor:
            results_list = list(executor.map(self._run_window_size, self.windowSizes))

        results_list.sort(key=lambda x: x[0])
        window_sizes = [item[0] for item in results_list]
        mean_psnrs = [item[1] for item in results_list]
        mean_ssims = [item[2] for item in results_list]

        fig_psnr, ax_psnr = plt.subplots()
        fig_ssim, ax_ssim = plt.subplots()

        ax_psnr.plot(window_sizes, mean_psnrs, marker='o', label='Mean PSNR')
        ax_ssim.plot(window_sizes, mean_ssims, marker='o', label='Mean SSIM')

        ax_psnr.set_title("Mean PSNR vs Window Size")
        ax_ssim.set_title("Mean SSIM vs Window Size")
        ax_psnr.set_xlabel("Window Size")
        ax_ssim.set_xlabel("Window Size")
        ax_psnr.set_ylabel("PSNR")
        ax_ssim.set_ylabel("SSIM")
        ax_psnr.legend()
        ax_ssim.legend()

        self.figures = [fig_psnr, fig_ssim]

    def show_plots(self):
        for fig in self.figures:
            fig.canvas.manager.set_window_title("Window Size Test Plots")
        plt.show()

    def save_plots(self, prefix="window_size"):
        for i, fig in enumerate(self.figures, 1):
            filename = os.path.join(self.plots_dir, f"{prefix}__{self.groundTruthName}_{i}.png")
            fig.savefig(filename)


class CompareVideos(ExperimentsSetup):
    def __init__(self, numberOfFrames, sparsityPercents=None, **kwargs):
        super().__init__(numberOfFrames, **kwargs)
        self.sparsityPercents = sparsityPercents or [10, 15, 20, 30, 50, 75]
        self.results = {}
        self.figures = []

    def run(self):
        max_processes = get_max_parallel_processes()
        all_ground_truths = self.SUPPORTED_GROUND_TRUTHS.keys()
        for gt_name in all_ground_truths:
            adaptive_args = [
                (sp, self.interpolMethod, self.numberOfFrames, gt_name, self.initialSampling)
                for sp in self.sparsityPercents
            ]
            with ProcessPoolExecutor(max_workers=min(len(adaptive_args), max_processes)) as executor:
                results_list = list(executor.map(run_adaptive_compare_helper, adaptive_args))

            self.results[gt_name] = {
                item[0]: {'mean_psnr': item[1], 'std_psnr': item[2], 'mean_ssim': item[3], 'std_ssim': item[4]}
                for item in results_list
            }

    def show_plots(self):
        fig_psnr, ax_psnr = plt.subplots()
        fig_ssim, ax_ssim = plt.subplots()
        for gt_name, data in self.results.items():
            sparsities = sorted(data.keys())
            mean_psnr = [data[sp]['mean_psnr'] for sp in sparsities]
            std_psnr = [data[sp]['std_psnr'] for sp in sparsities]
            mean_ssim = [data[sp]['mean_ssim'] for sp in sparsities]
            std_ssim = [data[sp]['std_ssim'] for sp in sparsities]

            ax_psnr.errorbar(sparsities, mean_psnr, yerr=std_psnr, label=gt_name, marker='o')
            ax_ssim.errorbar(sparsities, mean_ssim, yerr=std_ssim, label=gt_name, marker='o')

        ax_psnr.set_title("Mean PSNR vs Sparsity")
        ax_psnr.set_xlabel("Sparsity (%)")
        ax_psnr.set_ylabel("PSNR")
        ax_psnr.legend()
        ax_ssim.set_title("Mean SSIM vs Sparsity")
        ax_ssim.set_xlabel("Sparsity (%)")
        ax_ssim.set_ylabel("SSIM")
        ax_ssim.legend()
        self.figures = [fig_psnr, fig_ssim]
        for fig in self.figures:
            plt.show()

    def save_plots(self, prefix="compare_videos"):
        for i, fig in enumerate(self.figures, 1):
            filename = os.path.join(self.plots_dir, f"{prefix}_{i}.png")
            fig.savefig(filename)
