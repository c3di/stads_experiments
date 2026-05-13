import os

import cv2
import tifffile
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

EPS = 1e-8

def estimate_signal(stack):
    return np.mean(stack, axis=0).astype(np.float32)


def estimate_corr_length(img):
    img = img.astype(np.float32)

    f = np.fft.fft2(img)

    psd = np.abs(f) ** 2

    fy = np.fft.fftfreq(img.shape[0])
    fx = np.fft.fftfreq(img.shape[1])

    FX, FY = np.meshgrid(fx, fy)

    r2 = FX ** 2 + FY ** 2

    psd = psd + EPS
    psd /= np.sum(psd)

    spectral_width = np.sum(r2 * psd)

    corr_len = 1.0 / np.sqrt(spectral_width + EPS)

    return corr_len


def compute_stats(noiseVideoStack):
    stats = {}

    for t, stack in noiseVideoStack.items():

        signal = estimate_signal(stack)
        residuals = (stack.astype(np.float32) - signal[None])

        sigmas = []
        correlationLengths = []

        for frame in residuals:
            sigmas.append(np.std(frame))
            correlationLengths.append(estimate_corr_length(frame))

        stats[t] = {"signal": signal,
                    "noise": residuals[0],
                    "sigma": np.mean(sigmas),
                    "corr": np.mean(correlationLengths)}

    return stats


def gaussian_field(shape, corr):
    z = np.random.randn(*shape)

    fy = np.fft.fftfreq(shape[0])
    fx = np.fft.fftfreq(shape[1])

    FX, FY = np.meshgrid(fx, fy)

    f2 = FX ** 2 + FY ** 2

    lam = 1.0 / (corr + EPS)

    spectralFilter = 1.0 / (1.0 + f2 / (lam ** 2 + EPS))
    zf = np.fft.fft2(z)
    z = np.real(np.fft.ifft2(zf * spectralFilter))
    factor = max(int(corr / 2), 1)
    small = cv2.resize(z, (shape[1] // factor, shape[0] // factor), interpolation=cv2.INTER_AREA)
    z = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    z -= np.mean(z)
    z /= (np.std(z) + EPS)

    return z


def sigma_tail_model(t, a, alpha):
    return a * (t ** (-alpha))


def corr_tail_model(t, d, beta):
    return d * (t ** (-beta))


class SEMNoiseModel:

    def __init__(self, stats):

        self.stats = stats
        self.model = None

    def fit(self):

        stats = self.stats

        ts = np.array(sorted(stats.keys()), dtype=np.float32)
        u = np.log(ts + EPS)

        sigma = np.array([stats[t]["sigma"] for t in ts])
        corr = np.array([stats[t]["corr"] for t in ts])

        sigma_spline = CubicSpline(u, sigma)

        corr_spline = CubicSpline(u, corr)

        # sigma extrapolation
        p0_sigma = [sigma[0], 0.5, ]
        sigma_tail_params, _ = curve_fit(sigma_tail_model, ts, sigma, p0=p0_sigma, maxfev=40000)

        # corr extrapolation
        p0_corr = [corr[0], 0.5]
        corr_tail_params, _ = curve_fit(corr_tail_model, ts, corr, p0=p0_corr, maxfev=40000)

        self.model = {

            "sigma_spline":
                sigma_spline,

            "corr_spline":
                corr_spline,

            "sigma_tail":
                sigma_tail_params,

            "corr_tail":
                corr_tail_params,
        }

    def eval_model_at_t(self, t):

        model = self.model
        u = np.log(t + EPS)

        sigma_spline = model["sigma_spline"]
        corr_spline = model["corr_spline"]

        u_min = sigma_spline.x[0]
        u_max = sigma_spline.x[-1]


        if u_min <= u <= u_max:
            sigma = float(sigma_spline(u))
            corr = float(corr_spline(u))

        elif u > u_max:
            a, alpha = model["sigma_tail"]
            sigma = sigma_tail_model(t, a, alpha)
            d, beta = model["corr_tail"]
            corr = corr_tail_model(t, d, beta)

        else:
            sigma = float(sigma_spline(u_min))
            corr = float(corr_spline(u_min))

        sigma = max(sigma, 0)
        corr = max(corr, 0.5)

        return sigma, corr

    def generate_low_dwell_time_image(self, image_high, t_high, t_target):

        dtype = image_high.dtype

        image_high = image_high.astype(np.float32)

        if np.issubdtype(dtype, np.integer):

            input_max = np.iinfo(dtype).max
            signal = image_high / input_max

        else:
            input_max = 1.0
            signal = np.clip(image_high, 0, 1)

        sigma_h,_ = self.eval_model_at_t(t_high)
        sigma_t, corr_t = self.eval_model_at_t(t_target)

        delta_sigma2 = (sigma_t ** 2 - sigma_h ** 2)

        delta_sigma2 = np.maximum(delta_sigma2, 0)

        delta_sigma = np.sqrt(delta_sigma2)

        g = gaussian_field(signal.shape, corr_t)

        output = (signal + delta_sigma * g)

        output = np.clip(output, 0, 1)

        if np.issubdtype(dtype, np.integer):
            output = (output * input_max).astype(dtype)
        else:
            output = output.astype(np.float32)

        return output


    def visualize_empirical_noise(self):

        stats = self.stats

        ts = sorted(stats.keys())

        for t in ts:
            signal = stats[t]["signal"]

            noise = stats[t]["noise"]

            vmax = np.percentile(np.abs(noise), 99)
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # signal
            im0 = axs[0].imshow(signal, cmap='gray')
            axs[0].set_title(f"Signal Estimate\nDwell={t}")
            axs[0].axis('off')
            plt.colorbar(im0, ax=axs[0], fraction=0.046)

            # noise
            im1 = axs[1].imshow(noise, cmap='gray', vmin=-vmax, vmax=vmax)
            axs[1].set_title(f"Residual Noise\nDwell={t}")
            axs[1].axis('off')
            plt.colorbar(im1, ax=axs[1], fraction=0.046)
            plt.tight_layout()

            plt.show()

    # ========================================================
    # evaluate generator
    # ========================================================

    def evaluate_generator(self, allTiffs):

        stats = compute_stats(allTiffs)

        ts = np.array(sorted(stats.keys()))

        t_high = ts[-1]

        base = estimate_signal(allTiffs[t_high])

        sigma_emp = []
        sigma_gen = []

        corr_emp = []
        corr_gen = []

        for t in ts:
            sigma_emp.append(stats[t]["sigma"])

            corr_emp.append(stats[t]["corr"])

            generated = self.generate_low_dwell_time_image(base, t_high, t)

            residual = (generated.astype(np.float32) - base.astype(np.float32))

            sigma_gen.append(np.std(residual))

            corr_gen.append(estimate_corr_length(residual))

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # sigma
        axs[0].plot(ts, sigma_emp, 'o-', label='empirical')
        axs[0].plot(ts, sigma_gen, 's--', label='generated')
        axs[0].set_xscale('log')

        axs[0].set_title("Generated Sigma")
        axs[0].grid(True)
        axs[0].legend()

        # corr
        axs[1].plot(ts, corr_emp, 'o-', label='empirical')
        axs[1].plot(ts, corr_gen, 's--', label='generated')
        axs[1].set_xscale('log')
        axs[1].set_title("Generated Correlation")
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()


    def visualize_generated_noise(self, testImage, t_high, dwell_list):

        base = testImage.astype(np.float32)

        for t in dwell_list:
            generated = self.generate_low_dwell_time_image(testImage, t_high, t)
            noise = (generated.astype(np.float32) - base)

            vmax = np.percentile(np.abs(noise), 99)

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            im0 = axs[0].imshow(generated, cmap='gray')

            axs[0].set_title(f"Generated SEM\n{t_high} → {t}")

            axs[0].axis('off')

            plt.colorbar(im0, ax=axs[0], fraction=0.046)

            # noise
            im1 = axs[1].imshow(noise, cmap='gray', vmin=-vmax, vmax=vmax)

            axs[1].set_title(f"Generated Noise\nDwell={t}")

            axs[1].axis('off')

            plt.colorbar(im1, ax=axs[1], fraction=0.046)
            plt.tight_layout()
            plt.show()