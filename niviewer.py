import argparse
from pathlib import Path
import tkinter.filedialog as fd
import tkinter as tk

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector, PolygonSelector, Slider, Button, RadioButtons, TextBox
from matplotlib.patches import Ellipse, Polygon
from matplotlib import path as mpath
from matplotlib import cm, colors, ticker

from scipy.stats import skew, kurtosis

_tk_root = tk.Tk()
_tk_root.withdraw()

plt.rcParams.update({
    "figure.facecolor": "#2b2b2b",
    "axes.facecolor": "#2b2b2b",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "savefig.facecolor": "#2b2b2b",
    "savefig.edgecolor": "#2b2b2b",
    "font.family": "sans-serif",
    "font.sans-serif": ["CaskaydiaCove Nerd Font", "CaskaydiaCove NF", "Cascadia Code", "Consolas", "DejaVu Sans"],
})

ROI_COLORS = ["red", "lime", "dodgerblue", "orange", "magenta", "cyan", "yellow", "white"]


class NiftiEllipseROIViewer:
    def __init__(self, nii_path=None):
        self.nii_path = None
        self.data = None
        self.z = 1
        self.slice_idx = 0
        self.roi = None
        self.roi_type = "ellipse"
        self.rois = []
        self.poly_patch = None
        self.poly_verts = None
        self.mask = None

        self.auto_vmin, self.auto_vmax = 0, 1
        self.vmin, self.vmax = 0, 1

        self.fig = None
        self.ax = None
        self.cax = None
        self.im = None
        self.slider = None
        self.stats_texts = []
        self.cmap = "turbo"
        self.cbar = None
        self.ellipse_selector = None
        self.poly_selector = None
        self.vmin_text = None
        self.vmax_text = None
        self.hist_ax = None

        if nii_path is not None:
            self.load_file(nii_path)
        else:
            self._prompt_open()

    def _prompt_open(self):
        path = fd.askopenfilename(
            parent=_tk_root,
            title="Open NIfTI file",
            filetypes=[("NIfTI", "*.nii *.nii.gz"), ("All files", "*.*")],
        )
        if path:
            self.load_file(path)
        else:
            raise SystemExit("No file selected.")

    def load_file(self, path):
        self.nii_path = Path(path)
        self.img = nib.load(str(self.nii_path))
        data = self.img.get_fdata()

        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim != 3:
            raise RuntimeError("NIfTI must be 3D or 4D.")

        self.data = np.asarray(data, dtype=float)
        self.z = self.data.shape[2]
        self.slice_idx = self.z // 2
        self.roi = None
        self.roi_type = "ellipse"
        self.rois = []
        self.poly_patch = None
        self.poly_verts = None
        self.mask = None

        self.auto_vmin, self.auto_vmax = self._compute_volume_clim()
        self.vmin, self.vmax = self.auto_vmin, self.auto_vmax

    def open_file(self, event=None):
        path = fd.askopenfilename(
            parent=_tk_root,
            title="Open NIfTI file",
            filetypes=[("NIfTI", "*.nii *.nii.gz"), ("All files", "*.*")],
        )
        if not path:
            return
        self.load_file(path)
        self._reload_ui()

    def _reload_ui(self):
        # Disconnect old selectors entirely
        if self.ellipse_selector is not None:
            self.ellipse_selector.set_active(False)
            self.ellipse_selector.disconnect_events()
            self.ellipse_selector = None
        if self.poly_selector is not None:
            self.poly_selector.set_active(False)
            self.poly_selector.disconnect_events()
            self.poly_selector = None

        # Remove ALL patches from the image axes
        for p in self.ax.patches:
            p.remove()
        self.rois.clear()
        self.roi = None
        self.mask = None
        self.poly_verts = None
        if self.poly_patch is not None:
            self.poly_patch.remove()
            self.poly_patch = None

        # Update slider range without firing callbacks
        self.slider.eventson = False
        self.slider.valmin = 0
        self.slider.valmax = self.z - 1
        self.slider.ax.set_ylim(0, self.z - 1)
        self.slider.eventson = True
        self.slider.set_val(self.z // 2)

        # Update image data
        sl = self.get_slice()
        self.im.set_data(sl)
        self.im.set_clim(self.vmin, self.vmax)

        # Update title
        self.ax.set_title(
            f"{self.nii_path.name} | Slice {self.slice_idx + 1}/{self.z} | LUT: {self.cmap}",
            color="white",
        )

        # Update text boxes
        self.vmin_text.set_val(f"{self.vmin:.6g}")
        self.vmax_text.set_val(f"{self.vmax:.6g}")

        # Update colorbar
        self.refresh_colorbar(sl)

        # Reset stats and histogram
        self.compute_stats()
        self._update_histogram()

        # Create fresh selectors (no stale artists)
        self.ellipse_selector = EllipseSelector(
            self.ax, self.on_select,
            useblit=True,
            props=dict(edgecolor="yellow", facecolor="none", linewidth=1.5, alpha=0.8),
            interactive=True, drag_from_anywhere=True,
        )
        self.poly_selector = PolygonSelector(
            self.ax, self.on_poly_select,
            useblit=True,
            props=dict(color="cyan", linewidth=1.5, alpha=0.8),
        )
        self.poly_selector.set_active(False)

        self.fig.canvas.draw_idle()

    def _compute_volume_clim(self):
        finite = self.data[np.isfinite(self.data)]
        if finite.size == 0:
            return 0, 1
        vmin, vmax = np.percentile(finite, [2, 98])
        if vmin == vmax:
            vmax = vmin + 1
        return float(vmin), float(vmax)

    def get_slice(self):
        return self.data[:, :, self.slice_idx].T

    def current_clim(self):
        return self.vmin, self.vmax

    def _fmt(self, x, pos=None):
        return f"{x:.1e}" if x != 0 and abs(x) < 0.1 else f"{x:g}"

    def refresh_colorbar(self, sl=None):
        if sl is None:
            sl = self.get_slice()
        vmin, vmax = self.current_clim()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        sm.set_array([])
        if self.cbar is None:
            self.cbar = self.fig.colorbar(sm, cax=self.cax)
        else:
            self.cbar.mappable = sm
            self.cbar.update_normal(sm)
        self.cbar.set_label("Intensity", color="white")
        self.cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(self._fmt))
        self.cbar.ax.tick_params(colors="white")
        self.cbar.outline.set_edgecolor("white")
        for t in self.cbar.ax.get_yticklabels():
            t.set_color("white")

    def update_image(self):
        sl = self.get_slice()
        vmin, vmax = self.current_clim()
        self.im.set_data(sl)
        self.im.set_cmap(self.cmap)
        self.im.set_clim(vmin, vmax)
        self.ax.set_title(
            f"{self.nii_path.name} | Slice {self.slice_idx + 1}/{self.z} | LUT: {self.cmap}",
            color="white",
        )
        self.refresh_colorbar(sl)
        self.fig.canvas.draw_idle()

    def update_slice(self):
        self.im.set_data(self.get_slice())
        self.ax.set_title(
            f"{self.nii_path.name} | Slice {self.slice_idx + 1}/{self.z} | LUT: {self.cmap}",
            color="white",
        )
        self.fig.canvas.draw_idle()

    def on_slider(self, val):
        self.slice_idx = int(self.slider.val)
        self.clear_roi(draw=False)
        self.update_slice()

    def on_scroll(self, event):
        if self.slider is None or event.inaxes != self.ax:
            return
        step = 1 if event.button == "up" else -1
        self.slice_idx = max(0, min(self.z - 1, self.slice_idx + step))
        self.slider.set_val(self.slice_idx)

    def on_cmap(self, label):
        self.cmap = {"Gray": "gray", "Rainbow": "turbo", "Jet": "jet"}[label]
        self.update_image()

    def _hide_selector_artist(self, selector):
        for attr in ("_selection_artist", "to_draw", "line"):
            artist = getattr(selector, attr, None)
            if artist is not None:
                artist.set_visible(False)

    def clear_roi(self, draw=True):
        for roi in self.rois:
            roi["patch"].remove()
        self.rois.clear()
        self.roi = None
        self.mask = None
        self.poly_verts = None

        if self.poly_patch is not None:
            self.poly_patch.remove()
            self.poly_patch = None

        if self.ellipse_selector is not None:
            self._hide_selector_artist(self.ellipse_selector)
        if self.poly_selector is not None:
            self._hide_selector_artist(self.poly_selector)

        self.compute_stats()
        if draw and self.fig is not None:
            self.fig.canvas.draw_idle()

    def on_select(self, eclick, erelease):
        x0, y0, x1, y1 = eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return

        xc, yc = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        a, b = abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0
        if a <= 0 or b <= 0:
            return

        params = (xc, yc, a, b)
        self.roi = params
        self.roi_type = "ellipse"

        if self.rois:
            last = self.rois[-1]
            last["params"] = params
            last["patch"].remove()
            color = last["color"]
            patch = Ellipse(
                (xc, yc), 2 * a, 2 * b,
                angle=0, fill=False, edgecolor=color, linewidth=2,
            )
            self.ax.add_patch(patch)
            last["patch"] = patch
        else:
            color = ROI_COLORS[0]
            patch = Ellipse(
                (xc, yc), 2 * a, 2 * b,
                angle=0, fill=False, edgecolor=color, linewidth=2,
            )
            self.ax.add_patch(patch)
            self.rois.append({"params": params, "patch": patch, "color": color})

        self.compute_stats()
        self.fig.canvas.draw_idle()

    def duplicate_roi(self, event=None):
        if not self.rois:
            return
        xc, yc, a, b = self.rois[-1]["params"]
        xc += 20
        yc += 20
        color = ROI_COLORS[len(self.rois) % len(ROI_COLORS)]
        patch = Ellipse(
            (xc, yc), 2 * a, 2 * b,
            angle=0, fill=False, edgecolor=color, linewidth=2,
        )
        self.ax.add_patch(patch)
        entry = {"params": (xc, yc, a, b), "patch": patch, "color": color}
        self.rois.append(entry)
        self.roi = (xc, yc, a, b)
        self.roi_type = "ellipse"
        self.compute_stats()
        self.fig.canvas.draw_idle()

    def on_poly_select(self, verts):
        if len(verts) < 3:
            if self.poly_patch is not None:
                self.poly_patch.remove()
                self.poly_patch = None
            self.poly_verts = None
            self.roi = None
            self.mask = None
            self.compute_stats()
            self.fig.canvas.draw_idle()
            return

        self.poly_verts = verts
        self.roi = ("polygon", verts)
        self.roi_type = "polygon"

        if self.poly_patch is not None:
            self.poly_patch.remove()

        self.poly_patch = Polygon(
            verts, fill=False, edgecolor="red", linewidth=2,
        )
        self.ax.add_patch(self.poly_patch)
        self.compute_stats()
        self.fig.canvas.draw_idle()

    def _stats_for_mask(self, sl, mask, label):
        vals = sl[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return f"{label}: Empty"

        n = vals.size
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0
        vmin = vals.min()
        vmax = vals.max()
        median = np.median(vals)

        p10, p25, p75, p90 = np.percentile(vals, [10, 25, 75, 90])
        iqr = p75 - p25

        return (
            f"{label}  n={n}  mean={mean:.6g}  median={median:.6g}  std={std:.6g}\n"
            f"  min={vmin:.6g}  max={vmax:.6g}  P10={p10:.6g}  P90={p90:.6g}  IQR={iqr:.6g}"
        )

    def compute_stats(self):
        sl = self.get_slice()
        ny, nx = sl.shape

        for t in self.stats_texts:
            t.remove()
        self.stats_texts.clear()

        x_start = 0.02
        y_pos = 0.98
        for i, roi in enumerate(self.rois):
            xc, yc, a, b = roi["params"]
            yy, xx = np.mgrid[:ny, :nx]
            mask = ((xx - xc) / a) ** 2 + ((yy - yc) / b) ** 2 <= 1.0
            roi["mask"] = mask
            txt = self._stats_for_mask(sl, mask, f"ROI {i+1} [{roi['color'][0].upper()}]")
            t = self.ax.text(
                x_start, y_pos, txt,
                transform=self.ax.transAxes,
                va="top", ha="left",
                fontsize=8, color=roi["color"],
                bbox=dict(facecolor="#444444", alpha=0.75, edgecolor=roi["color"], pad=2),
            )
            self.stats_texts.append(t)
            x_start += 0.33

        if self.poly_verts is not None and len(self.poly_verts) >= 3:
            path = mpath.Path(self.poly_verts)
            yy, xx = np.mgrid[:ny, :nx]
            points = np.column_stack([xx.ravel(), yy.ravel()])
            mask = path.contains_points(points).reshape(ny, nx)
            self.mask = mask
            txt = self._stats_for_mask(sl, mask, "Polygon")
            t = self.ax.text(
                x_start, y_pos, txt,
                transform=self.ax.transAxes,
                va="top", ha="left",
                fontsize=8, color="red",
                bbox=dict(facecolor="#444444", alpha=0.75, edgecolor="red", pad=2),
            )
            self.stats_texts.append(t)

        if not self.stats_texts:
            t = self.ax.text(
                0.02, 0.98, "No ROI",
                transform=self.ax.transAxes,
                va="top", ha="left",
                fontsize=8, color="white",
            )
            self.stats_texts.append(t)

        self._update_histogram()

    def _update_histogram(self):
        if self.hist_ax is None:
            return
        self.hist_ax.clear()
        sl = self.get_slice()

        if not self.rois and self.mask is None:
            self.hist_ax.text(0.5, 0.5, "No ROI", transform=self.hist_ax.transAxes,
                              ha="center", va="center", color="white", fontsize=10)
        else:
            for roi in self.rois:
                m = roi.get("mask")
                if m is None:
                    continue
                vals = sl[m]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                hist, bins = np.histogram(vals, bins=64)
                centers = (bins[:-1] + bins[1:]) / 2
                self.hist_ax.bar(centers, hist, width=np.diff(bins), alpha=0.5,
                                 color=roi["color"], edgecolor="none")

            if self.mask is not None:
                vals = sl[self.mask]
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    hist, bins = np.histogram(vals, bins=64)
                    centers = (bins[:-1] + bins[1:]) / 2
                    self.hist_ax.bar(centers, hist, width=np.diff(bins), alpha=0.5,
                                     color="red", edgecolor="none")

        self.hist_ax.set_title("ROI Histogram", color="white", fontsize=10)
        self.hist_ax.tick_params(colors="white", labelsize=8)
        self.hist_ax.set_xlabel("Intensity", color="white", fontsize=8)
        self.hist_ax.set_ylabel("Count", color="white", fontsize=8)
        self.hist_ax.set_facecolor("#2b2b2b")
        for spine in self.hist_ax.spines.values():
            spine.set_color("white")

    def set_windowing(self, vmin, vmax):
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.update_image()

    def auto_windowing(self, event=None):
        self.vmin, self.vmax = self.auto_vmin, self.auto_vmax
        if self.vmin_text is not None:
            self.vmin_text.set_val(f"{self.auto_vmin:.6g}")
        if self.vmax_text is not None:
            self.vmax_text.set_val(f"{self.auto_vmax:.6g}")
        self.update_image()

    def on_vmin_submit(self, text):
        try:
            val = float(text)
            self.set_windowing(val, self.vmax)
        except ValueError:
            pass

    def on_vmax_submit(self, text):
        try:
            val = float(text)
            self.set_windowing(self.vmin, val)
        except ValueError:
            pass

    def on_roi_type(self, label):
        self.clear_roi()
        self.roi_type = {"Ellipse": "ellipse", "Polygon": "polygon"}[label]
        if self.roi_type == "ellipse":
            self.ellipse_selector.set_active(True)
            self.poly_selector.set_active(False)
        else:
            self.ellipse_selector.set_active(False)
            self.poly_selector.set_active(True)

    def save_mask(self, event=None):
        if not self.rois and self.mask is None:
            self.compute_stats()
            return

        base = self.nii_path.stem.replace(".nii", "")
        n = 0
        for i, roi in enumerate(self.rois):
            m = roi.get("mask")
            if m is not None:
                out = self.nii_path.with_name(f"{base}_roi{i+1}_slice{self.slice_idx + 1}.png")
                plt.imsave(str(out), m.T.astype(float), cmap="gray")
                n += 1

        if self.mask is not None:
            out = self.nii_path.with_name(f"{base}_poly_slice{self.slice_idx + 1}.png")
            plt.imsave(str(out), self.mask.T.astype(float), cmap="gray")
            n += 1

        for t in self.stats_texts:
            t.remove()
        self.stats_texts.clear()
        t = self.ax.text(
            0.02, 0.98, f"Saved {n} mask(s)",
            transform=self.ax.transAxes,
            va="top", ha="left",
            fontsize=9, color="lime",
        )
        self.stats_texts.append(t)
        self.fig.canvas.draw_idle()

    def launch(self):
        self.fig = plt.figure(figsize=(11, 8), facecolor="#2b2b2b")

        self.cax = self.fig.add_axes([0.92, 0.02, 0.02, 0.94], facecolor="#2b2b2b")
        self.ax = self.fig.add_axes([0.18, 0.02, 0.62, 0.94], facecolor="#2b2b2b")

        sl = self.get_slice()
        vmin, vmax = self.current_clim()
        self.im = self.ax.imshow(sl, cmap=self.cmap, origin="lower", vmin=vmin, vmax=vmax)
        self.ax.set_aspect("equal")
        self.ax.tick_params(colors="white")
        self.ax.set_title(
            f"{self.nii_path.name} | Slice {self.slice_idx + 1}/{self.z} | LUT: {self.cmap}",
            color="white",
        )

        self.stats_texts = []
        self.compute_stats()

        self.refresh_colorbar(sl)

        ax_load = self.fig.add_axes([0.02, 0.88, 0.12, 0.04], facecolor="#2b2b2b")
        ax_vmin = self.fig.add_axes([0.02, 0.80, 0.12, 0.04], facecolor="#444444")
        ax_vmax = self.fig.add_axes([0.02, 0.74, 0.12, 0.04], facecolor="#444444")
        ax_auto = self.fig.add_axes([0.02, 0.68, 0.12, 0.04], facecolor="#2b2b2b")

        ax_radio = self.fig.add_axes([0.02, 0.46, 0.12, 0.18], facecolor="#2b2b2b")
        ax_roi_type = self.fig.add_axes([0.02, 0.24, 0.12, 0.18], facecolor="#2b2b2b")
        ax_hist = self.fig.add_axes([0.02, 0.02, 0.12, 0.18], facecolor="#2b2b2b")

        ax_slider = self.fig.add_axes([0.82, 0.02, 0.03, 0.94], facecolor="#2b2b2b")

        ax_clear = self.fig.add_axes([0.88, 0.02, 0.03, 0.04], facecolor="#2b2b2b")
        ax_save = self.fig.add_axes([0.88, 0.07, 0.03, 0.04], facecolor="#2b2b2b")
        ax_dup = self.fig.add_axes([0.88, 0.12, 0.03, 0.04], facecolor="#2b2b2b")

        btn_load = Button(ax_load, "Load", color="#444444", hovercolor="#666666")
        btn_load.on_clicked(self.open_file)

        self.vmin_text = TextBox(ax_vmin, "Min", initial=f"{self.vmin:.6g}", color="#444444", hovercolor="#555555")
        self.vmin_text.on_submit(self.on_vmin_submit)

        self.vmax_text = TextBox(ax_vmax, "Max", initial=f"{self.vmax:.6g}", color="#444444", hovercolor="#555555")
        self.vmax_text.on_submit(self.on_vmax_submit)

        btn_auto = Button(ax_auto, "Auto", color="#444444", hovercolor="#666666")
        btn_auto.on_clicked(self.auto_windowing)

        btn_clear = Button(ax_clear, "C", color="#444444", hovercolor="#666666")
        btn_clear.on_clicked(lambda event: self.clear_roi())

        btn_save = Button(ax_save, "S", color="#444444", hovercolor="#666666")
        btn_save.on_clicked(self.save_mask)

        btn_dup = Button(ax_dup, "D", color="#444444", hovercolor="#666666")
        btn_dup.on_clicked(self.duplicate_roi)

        self.slider = Slider(
            ax_slider, "Slice", 0, self.z - 1,
            valinit=self.slice_idx, valstep=1, color="#4c8eda",
            orientation="vertical",
        )
        self.slider.on_changed(self.on_slider)

        radio = RadioButtons(ax_radio, ("Gray", "Rainbow", "Jet"), active=2)
        radio.on_clicked(self.on_cmap)
        for txt in radio.labels:
            txt.set_color("white")

        radio_roi = RadioButtons(ax_roi_type, ("Ellipse", "Polygon"), active=0)
        radio_roi.on_clicked(self.on_roi_type)
        for txt in radio_roi.labels:
            txt.set_color("white")

        self.hist_ax = ax_hist
        self.hist_ax.set_title("ROI Histogram", color="white", fontsize=9)
        self.hist_ax.tick_params(colors="white", labelsize=7)
        self.hist_ax.set_xlabel("Intensity", color="white", fontsize=7)
        self.hist_ax.set_ylabel("Count", color="white", fontsize=7)
        self.hist_ax.set_facecolor("#2b2b2b")
        for spine in self.hist_ax.spines.values():
            spine.set_color("white")

        self.ellipse_selector = EllipseSelector(
            self.ax, self.on_select,
            useblit=True,
            props=dict(edgecolor="yellow", facecolor="none", linewidth=1.5, alpha=0.8),
            interactive=True, drag_from_anywhere=True,
        )

        self.poly_selector = PolygonSelector(
            self.ax, self.on_poly_select,
            useblit=True,
            props=dict(color="cyan", linewidth=1.5, alpha=0.8),
        )
        self.poly_selector.set_active(False)

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.compute_stats()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Lightweight ROI viewer for NIfTI files")
    parser.add_argument("input", nargs="?", default=None, help="Path to .nii or .nii.gz (optional; opens file dialog if omitted)")
    args = parser.parse_args()
    viewer = NiftiEllipseROIViewer(args.input)
    viewer.launch()


if __name__ == "__main__":
    main()
