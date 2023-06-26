#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  visualizer.py

import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as Tk

matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import sys
import os

from radiotools import helper, coordinatesystems
from radiotools.atmosphere import models
from geoceLDF import LDF
from scipy.interpolate import interp1d

from skimage.draw import polygon
from skimage.draw import rectangle
from skimage.draw import disk

xmax_DF = pd.read_csv("./Xmax.dat")
XmaxInterp = interp1d(
    10 ** xmax_DF.energy_log10.values, xmax_DF.xmax_gcm2.values, kind="slinear"
)

# triangle
def CreateArray_triangle(side=7, stations_distance_km=1.500, array_layout="square"):
    x_stations_distance_km = stations_distance_km
    y_stations_distance_km = (
        stations_distance_km ** 2 - (stations_distance_km / 2) ** 2
    ) ** (1 / 2)
    # create station position dictionary
    station_dict = {}
    # position of the first station
    x_start = 0
    y_start = 0
    x = x_start
    y = y_start
    z = 1.5
    stations_in_x = 0
    if (array_layout == "hexagon") or (array_layout == "octagon"):
        extra = 0
        x_offset = 0
        last_time = None
        if side % 2 == True:
            odd = 1
        else:
            odd = 0
        if array_layout == "hexagon":
            skip_side = 0
        else:
            skip_side = 1
        station_number = 1
        for row in range(1, int(2 * side + (side - 2) * skip_side + odd * skip_side)):
            for s in range(int(side + extra)):
                station_dict["LS_" + str(station_number)] = {
                    "x": x - x_offset,
                    "y": y,
                    "z": z,
                }
                station_number += 1
                x = x + stations_distance_km
            y = y + y_stations_distance_km
            x = x_start
            if row < side:
                x_offset = x_offset + x_stations_distance_km / 2
                extra += 1
            elif (row >= side) and (row <= (side + (side - 2))) and skip_side != 0:
                if last_time == None:
                    extra -= 1
                    x_offset = x_offset - x_stations_distance_km / 2
                    last_time = "decrease"
                elif last_time == "increase":
                    extra -= 1
                    x_offset = x_offset - x_stations_distance_km / 2
                    last_time = "decrease"
                elif last_time == "decrease":
                    extra += 1
                    x_offset = x_offset + x_stations_distance_km / 2
                    #   x_offset = x_offset + x_stations_distance_km/2
                    last_time = "increase"
            elif (row > (side + (side - 2) * skip_side)) or (
                row >= side and skip_side == 0
            ):
                x_offset = x_offset - x_stations_distance_km / 2
                extra -= 1
    elif array_layout == "square":
        start = (x_start, y_start)
        extent = (side, side)
        rr, cc = rectangle(start, extent=extent)
        rr = rr.flatten()
        cc = cc.flatten()
        for i, r in enumerate(rr):
            if ((cc[i] - np.min(cc)) % 2) == False:
                station_dict["LS_" + str(i + 1)] = {
                    "x": rr[i] * stations_distance_km,
                    "y": cc[i] * stations_distance_km,
                    "z": z,
                }
            if (((cc[i] - np.min(cc)) % 2) == True) and rr[i] != np.max(rr):
                station_dict["LS_" + str(i + 1)] = {
                    "x": rr[i] * stations_distance_km + stations_distance_km / 2,
                    "y": cc[i] * stations_distance_km,
                    "z": z,
                }
    elif array_layout == "circle":
        shape = (side + 3, side + 3)
        rr, cc = disk((side / 2 + 1, side / 2 + 1), side / 2 + 1)
        r_min = np.min(rr)
        for i, r in enumerate(rr):
            if ((cc[i] - np.min(cc)) % 2) == False:
                station_dict["LS_" + str(i + 1)] = {
                    "x": rr[i] * stations_distance_km,
                    "y": cc[i] * stations_distance_km,
                    "z": z,
                }
            if (((cc[i] - np.min(cc)) % 2) == True) and rr[i] != np.max(rr):
                station_dict["LS_" + str(i + 1)] = {
                    "x": rr[i] * stations_distance_km + stations_distance_km / 2,
                    "y": cc[i] * stations_distance_km,
                    "z": z,
                }
        # for i, r in enumerate(rr):
        # offset = stations_distance_km*(r-1-r_min)*0
        # if r in rows[1::2]:
        # station_dict['LS_'+str(i+1)] = {
        # 'x':cc[i]*stations_distance_km+stations_distance_km/2+ offset,
        # 'y':rr[i]*y_stations_distance_km,'z':z}
        # else:
        # station_dict['LS_'+str(i+1)] = {'x':cc[i]*stations_distance_km+offset,'y':rr[i]*y_stations_distance_km,'z':z}
    return station_dict


# square
def CreateArray_square(side=7, stations_distance_km=1500, array_layout="square"):
    row_num = 2 * (side - 1) + 3
    col_num = side + 2 * (side - 1) + 2
    v1 = (side, 1)
    v2 = v1[0] - (side - 1), v1[1] + (side - 1)
    v3 = v2[0], v2[1] + side - 1
    r = []
    c = []
    if array_layout == "hexagon":
        v4 = v3[0] + (side - 1), v3[1] + (side - 1)
        v6 = v4[0] + (side - 1), v4[1] - (side - 1)
        v7 = v6[0], v6[1] - (side - 1)
        for v in [v1, v2, v3, v4, v6, v7]:
            r.append(v[0])
            c.append(v[1])
        rr, cc = polygon(c, r)
    elif array_layout == "octagon":
        v2 = v1[0] - (side - 1), v1[1] + (side - 1)
        v3 = v2[0], v2[1] + side - 1
        v4 = v3[0] + (side - 1), v3[1] + (side - 1)
        v5 = v4[0] + side - 1, v4[1]
        v6 = v5[0] + (side - 1), v5[1] - (side - 1)
        v7 = v6[0], v6[1] - (side - 1)
        v8 = v7[0] - (side - 1), v7[1] - (side - 1)
        for v in [v1, v2, v3, v4, v5, v6, v7, v8]:
            r.append(v[0])
            c.append(v[1])
        #
        rr, cc = polygon(r, c)
    if array_layout == "square":
        start = (0, 0)
        extent = (side, side)
        rr, cc = rectangle(start, extent=extent)
        rr = rr.flatten()
        cc = cc.flatten()
    if array_layout == "circle":
        shape = (side + 3, side + 3)
        rr, cc = disk((side / 2 + 1, side / 2 + 1), side / 2 + 1)
    #
    z = 1.500
    station_dict = {}
    for i in range(len(rr)):
        station_dict["LS_" + str(i + 1)] = {
            "x": rr[i] * stations_distance_km,
            "y": cc[i] * stations_distance_km,
            "z": z,
        }
    # img[rr, cc] = 1
    # img
    # np.vstack(( np.arange(-1,row_num), np.vstack((np.arange(col_num), img)).T )).T
    return station_dict


def Create_array_layout(
    side=7,
    stations_distance_km=1500,
    array_layout="octagon",
    stations_layout="triangle",
):
    if stations_layout == "triangle":
        return CreateArray_triangle(
            side=side,
            stations_distance_km=stations_distance_km,
            array_layout=array_layout,
        )
    elif stations_layout == "square":
        return CreateArray_square(
            side=side,
            stations_distance_km=stations_distance_km,
            array_layout=array_layout,
        )


# calculate energy fluences
def Calculate_energy_fluences(
    station_dict,
    energy,
    azimuth,
    zenith,
    core_x_km=0,
    core_y_km=0,
    B_earthVector=helper.get_magnetic_field_vector(site="auger"),
):
    km2meters = 1000
    core_x_m = core_x_km * km2meters
    core_y_m = core_y_km * km2meters
    # energy = 50
    energy_cr = energy * 1e18
    # observer height
    obsheight = 1564.0
    # magnetic field vector
    B_earth = np.sqrt(np.sum(B_earthVector ** 2))
    alpha = helper.get_sine_angle_to_lorentzforce(
        zenith, azimuth, magnetic_field_vector=B_earthVector
    )
    # calculate radiation energy and xmax and dxmax
    Erad = (
        1000000
        * 15.8
        * (np.sin(alpha) * (energy_cr / 10 ** 18) * (B_earth / 0.24)) ** 2
    )  # E_30_80MHz
    Xmax = XmaxInterp(energy_cr)
    atm = models.Atmosphere()
    dxmax = atm.get_distance_xmax(zenith, Xmax, obsheight)
    event = coordinatesystems.cstrafo(
        zenith, azimuth, magnetic_field_vector=B_earthVector, site=None
    )  # offline convention
    for key in station_dict:
        coordinates = np.array(
            [
                station_dict[key]["x"] * km2meters - core_x_m,
                station_dict[key]["y"] * km2meters - core_y_m,
                station_dict[key]["z"] * km2meters,
            ]
        )
        station_dict[key]["x_VBxVVB"], station_dict[key]["y_VBxVVB"], station_dict[key][
            "z_VBxVVB"
        ] = event.transform_to_vxB_vxvxB(coordinates, core=None)
        station_dict[key]["energy_fluence"], station_dict[key][
            "energy_fluence_VB"
        ], station_dict[key]["energy_fluence_VVB"], _, _ = LDF.LDF_geo_ce(
            station_dict[key]["x_VBxVVB"],
            station_dict[key]["y_VBxVVB"],
            Erad,
            dxmax,
            zenith,
            azimuth,
            core=np.array([0, 0]),
            obsheight=1564.0,
            magnetic_field_vector=B_earthVector,
        )  # 1564.0
    return station_dict


_VARS = {
    "window": False,
    "fig_agg": False,
    "pltFig": False,
    "fig_aggB": False,
    "pltFigB": False,
}


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=1)
    #
    def on_key_press(event):
        key_press_handler(event, canvas, toolbar)
        canvas.TKCanvas.mpl_connect("key_press_event", on_key_press)

    return figure_canvas_agg


class Toolbar(NavigationToolbar2Tk):
    # only display the buttons we need
    toolitems = [
        t
        for t in NavigationToolbar2Tk.toolitems
        if t[0] in ("Home", "Pan", "Zoom", "Save")
    ]
    # t[0] in ('Home', 'Pan', 'Zoom','Save')]
    #
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


# First the window layout
def_stations_layout = "square"
def_array_layout = "square"
def_stations_distance_km = "1.5"
def_z = "1.500"
def_num_side_stations = "10"
B_earthVector = helper.get_magnetic_field_vector(site="auger")

# emphasize font
font_emphasize = ("Helvetica", 14, "bold")

output_mag_field = sg.Text(
    "Magnetic field is set to: {}, {}, {}.".format(*B_earthVector)
)

array_list_column = [
    [sg.Text("CONTROL PANEL", font=font_emphasize)],
    [sg.HorizontalSeparator()],
    [
        sg.Text("Magnetic field vector (x,y,z) [Gauss]: "),
        sg.In(
            size=(8, 1),
            default_text=B_earthVector[0],
            enable_events=True,
            key="-MAG_VECTOR_X-",
        ),
        sg.In(
            size=(8, 1),
            default_text=B_earthVector[1],
            enable_events=True,
            key="-MAG_VECTOR_Y-",
        ),
        sg.In(
            size=(8, 1),
            default_text=B_earthVector[2],
            enable_events=True,
            key="-MAG_VECTOR_Z-",
        ),
    ],
    [sg.Button("SET", key="-SET_MAGNETIC_VECTOR-"), output_mag_field],
    [sg.HorizontalSeparator()],
    [
        sg.Text("Number of stations on the side"),
        sg.In(
            size=(4, 1),
            default_text=def_num_side_stations,
            enable_events=True,
            key="-STATIONS_ON_SIDE-",
        ),
    ],
    [
        sg.Text("Station distance [km]" + " " * 14),
        sg.In(
            size=(4, 1),
            default_text=def_stations_distance_km,
            enable_events=True,
            key="-STATION_DISTANCE-",
        ),
    ],
    [
        sg.Text("Stations layout" + " " * 23),
        sg.Combo(
            ["triangle", "square"],
            default_value=def_stations_layout,
            key="-STATIONS_LAYOUT-",
        ),
    ],
    [
        sg.Text("Array layout" + " " * 27),
        sg.Combo(
            ["circle", "square", "hexagon", "octagon"],
            default_value=def_stations_layout,
            key="-ARRAY_LAYOUT-",
        ),
    ],
    [sg.Button("CREATE ARRAY", key="-CREATE_ARRAY-")],
    [sg.HorizontalSeparator()],
    [
        sg.Text("Load custom array"),
        sg.In(size=(34, 1)),
        sg.FileBrowse("browse", key="-BROWSE-"),
    ],
    [sg.Button("LOAD", key="-LOAD_CUSTOM_ARRAY-")],
]


station_dict = Create_array_layout(
    side=float(def_num_side_stations),
    stations_distance_km=float(def_stations_distance_km),
    array_layout=def_array_layout,
    stations_layout=def_stations_layout,
)
station_DF = pd.DataFrame(station_dict).T

def_energy = 10
def_azimuth = 45
def_zenith = 0
def_core_x_km = station_DF.x.mean()
def_core_y_km = station_DF.y.mean()
shower_list_column = [
    [
        sg.Text("Energy [EeV]\t"),
        sg.Slider(
            range=(1, 94),
            resolution=1,
            size=(30, 20),
            default_value=def_energy,
            orientation="h",
            key="-ENERGY-",
        ),
    ],
    [
        sg.Text("Zenith [degs]\t"),
        sg.Slider(
            range=(25, 84),
            resolution=1,
            size=(30, 20),
            default_value=def_zenith,
            orientation="h",
            key="-ZENITH-",
        ),
    ],
    [
        sg.Text("Azimuth [degs]\t"),
        sg.Slider(
            range=(0, 360),
            resolution=1,
            size=(30, 20),
            default_value=def_azimuth,
            orientation="h",
            key="-AZIMUTH-",
        ),
    ],
    [
        sg.Text("Shower core x [km]"),
        sg.In(
            size=(5, 1), default_text=def_core_x_km, enable_events=True, key="-CORE_X-"
        ),
    ],
    [
        sg.Text("Shower core y [km]"),
        sg.In(
            size=(5, 1), default_text=def_core_y_km, enable_events=True, key="-CORE_Y-"
        ),
    ],
    [
        sg.Text("Shower type:\t"),
        sg.Radio("Hadronic", "PRIMARY1", default=True, key="-IN2-"),
        #      sg.Radio("Neutrino shower [not implemented]", "PRIMARY1", default=False)
    ],
]

submit_array_column = [
    [sg.HorizontalSeparator()],
    [
        sg.Text("Marker size\t"),
        sg.Slider(
            range=(0.2, 50),
            resolution=0.2,
            size=(30, 20),
            default_value="20",
            orientation="h",
            key="-MARKER_SIZE-",
        ),
    ],
    [
        sg.Checkbox(
            "Show magnetic field direction", default=False, key="-SHOW_MAG_VECTOR-"
        )
    ],
    [sg.Button("UPDATE", key="-UPDATE-")],
    [sg.HorizontalSeparator()],
]
values = {
    "-MARKER_SIZE-": "20",
    "-MAG_VECTOR_X-": B_earthVector[0],
    "-MAG_VECTOR_Y-": B_earthVector[1],
    "-SHOW_MAG_VECTOR-": False,
}

submit_shower_column = [
    [sg.Button("SUBMIT", key="-SUBMIT-")],
    [sg.HorizontalSeparator()],
    [sg.Button("HELP", key="-HELP-")],
]

# For now will only show the name of the file that was chosen
output = sg.Text(
    "Energy: {} [EeV] Azimuth: {} [degs] Zenith: {} [degs] Core X: {:.2f} [km] Core Y:{:.2f} [km]".format(
        def_energy, def_azimuth, def_zenith, def_core_x_km, def_core_y_km
    ),
    font=font_emphasize,
    text_color="black",
    background_color="white",
)

output2 = sg.Text(
    "Total number of stations in the array: {}".format(len(station_dict)),
    font=font_emphasize,
    text_color="black",
    background_color="white",
    justification="center",
)

w_1, h_1 = figsize_1 = (7, 5)     # figure size
w_2, h_2 = figsize_2 = (4.5, 4.5)
dpi = 100 
size_1 = (w_1*dpi, h_1*dpi)
size_2 = (w_2*dpi, h_2*dpi)

image_viewer_column = [
    [output],
    [output2],
    [sg.Canvas(size=size_1, key="-CANVAS-")],
    [sg.Canvas(key="-CANVAS_CONTROL-")],
    #    [sg.Canvas(key="-CANVAS_B-")]
]


def fluence_def_string(station_DF):
    total_stations = len(station_DF.index.values)
    try:
        number_of_triggered_stations = len(
            station_DF[station_DF.energy_fluence.values > 5]
        )
    except:
        number_of_triggered_stations = 0
    return "Number of stations with energy fluence >= 5eV/m\u00b2: {} \nNumber of stations with energy fluence < 5eV/m\u00b2: {}".format(
        number_of_triggered_stations, total_stations - number_of_triggered_stations
    )


fluence_info = sg.Text(
    fluence_def_string(pd.DataFrame(station_dict).T),
    font=font_emphasize,
    text_color="black",
    background_color="white",
    pad=((2, 0), (5, 15)),
)
image_viewer_column2 = [
    [fluence_info],
    [sg.Canvas(size=size_2, key="-CANVAS_B-")],
    [sg.Canvas(key="-CANVAS_CONTROL_B-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(
            array_list_column
            + submit_array_column
            + shower_list_column
            + submit_shower_column,
            vertical_alignment="top",
        ),
        sg.VSeperator(),
        sg.Column(
            image_viewer_column,
            vertical_alignment="top",
            justification="center",
            expand_x=True,
        ),
        sg.VSeperator(),
        sg.Column(image_viewer_column2, vertical_alignment="top", pad=0),
    ]
]


########################################################################


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


cmap_original = plt.get_cmap("jet")
cmap = truncate_colormap(cmap_original, 0.2, 1)

left_A = 0.2
right_A = 0.9


def drawChart(station_DF, values=None):
    _VARS["pltFig"] = plt.figure(figsize=figsize_1)
    # plt.scatter(station_DF.x.values, station_DF.y.values)
    ax = _VARS["pltFig"].add_subplot(111)
    sc = ax.scatter(
        station_DF.x.values,
        station_DF.y.values,
        cmap=cmap,
        s=float(values["-MARKER_SIZE-"]),
        vmin=np.log10(5),
        vmax=4,
        c=np.zeros(station_DF.x.values.size),
    )
    cb = plt.colorbar(sc, ax=ax)
    ax.figure.set_size_inches(8, 6)
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    cb.set_label("log10(energy fluence/eV/m\u00b2)")
 #   _VARS["pltFig"].subplots_adjust(left=left_A, right=right_A)
    _VARS["fig_agg"] = draw_figure_w_toolbar(
        _VARS["window"]["-CANVAS-"].TKCanvas,
        _VARS["pltFig"],
        _VARS["window"]["-CANVAS_CONTROL-"].TKCanvas,
    )
    ax.axes.set_aspect("equal")


def updateChart(station_DF, values=None):
    _VARS["fig_agg"].get_tk_widget().forget()
    plt.cla()
    plt.clf()
    _VARS["pltFig"].clear()
    ax = _VARS["pltFig"].add_subplot(111)
    sc = ax.scatter(
        station_DF.x.values,
        station_DF.y.values,
        cmap=cmap,
        s=float(values["-MARKER_SIZE-"]),
        vmin=np.log10(5),
        vmax=4,
        c=np.zeros(station_DF.x.values.size),
    )
    cb = plt.colorbar(sc, ax=ax)
    ax.figure.set_size_inches(8, 6)
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    cb.set_label("log10(energy fluence/eV/m\u00b2)")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    length = np.abs(np.diff(xlim)[0]) / 2.5
    if values["-SHOW_MAG_VECTOR-"] == True:
        start_x = np.mean(station_DF.x.values)
        start_y = np.mean(station_DF.y.values)
        ax.arrow(
            start_x,
            start_y,
            float(values["-MAG_VECTOR_X-"]) * length * 4,
            float(values["-MAG_VECTOR_Y-"]) * length * 4,
            width=0.1,
            head_width=20 * (length / 250),
            head_length=20 * (length / 250),
            length_includes_head=False,
            color="black",
        )
    _VARS["pltFig"].subplots_adjust(left=left_A, right=right_A)
    _VARS["fig_agg"] = draw_figure_w_toolbar(
        _VARS["window"]["-CANVAS-"].TKCanvas,
        _VARS["pltFig"],
        _VARS["window"]["-CANVAS_CONTROL-"].TKCanvas,
    )
    ax.axes.set_aspect("equal")


def updateChart2(station_DF, values=None):
    _VARS["fig_agg"].get_tk_widget().forget()
    plt.cla()
    plt.clf()
    _VARS["pltFig"].clear()
    ax = _VARS["pltFig"].add_subplot(111)
    # to avoid zero in log
    e_values = station_DF.energy_fluence.values
    e_values[np.isnan(e_values)] = 1e-10
    e_values[e_values < 1e-10] = 1e-10
    sc = ax.scatter(
        station_DF.x.values,
        station_DF.y.values,
        c=np.log10(e_values),
        cmap=cmap,
        s=float(values["-MARKER_SIZE-"]),
        vmin=np.log10(5),
        vmax=4,
    )
    cb = plt.colorbar(sc, ax=ax, extend="max")
    ax.figure.set_size_inches(8, 6)
    ax.axes.set_aspect("equal")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    length = np.abs(np.diff(xlim)[0]) / 2.5
    c_x = float(values["-CORE_X-"])
    c_y = float(values["-CORE_Y-"])
    shower_ax_x, shower_ax_y = get_shower_axis(
        c_x=c_x, c_y=c_y, angle=values["-AZIMUTH-"], length=length
    )
    ax.plot(shower_ax_x, shower_ax_y, color="magenta", linewidth=3)
    ax.scatter(
        c_x,
        c_y,
        marker="x",
        color="magenta",
        s=float(values["-MARKER_SIZE-"] * 10),
        linewidths=3,
    )
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    cb.set_label("log10(energy fluence/eV/m\u00b2)")
    if values["-SHOW_MAG_VECTOR-"] == True:
        start_x = np.mean(station_DF.x.values)
        start_y = np.mean(station_DF.y.values)
        ax.arrow(
            start_x,
            start_y,
            float(values["-MAG_VECTOR_X-"]) * length * 4,
            float(values["-MAG_VECTOR_Y-"]) * length * 4,
            width=0.1,
            head_width=20 * (length / 250),
            head_length=20 * (length / 250),
            length_includes_head=False,
            color="black",
        )
    _VARS["pltFig"].subplots_adjust(left=left_A, right=right_A)
    #  ax2 = _VARS['pltFig'] .add_subplot(212)
    #  ax2.hist(station_DF.energy_fluence.values)
    _VARS["fig_agg"] = draw_figure_w_toolbar(
        _VARS["window"]["-CANVAS-"].TKCanvas,
        _VARS["pltFig"],
        _VARS["window"]["-CANVAS_CONTROL-"].TKCanvas,
    )


#  return start_x, start_y, float(values['-MAG_VECTOR_X-']), float(values['-MAG_VECTOR_Y-'])


def get_shower_axis(c_x=0, c_y=0, angle=0, length=1):
    x = 0
    y = 0
    x2 = np.cos(np.deg2rad(angle))
    y2 = np.sin(np.deg2rad(angle))
    xx = np.array([x, x2])
    yy = np.array([y, y2])
    xx[1] = xx[1] * length
    yy[1] = yy[1] * length
    return xx + c_x, yy + c_y


########################################################################

left_B = 0.2
bottom_B = 0.15
right_B = 0.9


def drawChartB(station_DF):
    _VARS["pltFigB"] = plt.figure(figsize=figsize_2)
    # plt.scatter(station_DF.x.values, station_DF.y.values)
    ax = _VARS["pltFigB"].add_subplot(111)
    ax.set_xlabel("energy fluence [eV/m\u00b2]")
    ax.set_ylabel("frequency")
    #  ax.hist([0,0])
    _VARS["pltFigB"].subplots_adjust(left=left_B, bottom=bottom_B, right=right_B)
    _VARS["fig_aggB"] = draw_figure_w_toolbar(
        _VARS["window"]["-CANVAS_B-"].TKCanvas,
        _VARS["pltFigB"],
        _VARS["window"]["-CANVAS_CONTROL_B-"].TKCanvas,
    )


def updateChartB(station_DF):
    _VARS["fig_aggB"].get_tk_widget().forget()
    plt.cla()
    plt.clf()
    _VARS["pltFigB"].clear()
    ax = _VARS["pltFigB"].add_subplot(111)
    if "energy_fluence" in station_DF:
        ax.hist(
            station_DF.energy_fluence[(station_DF.energy_fluence.values >= 5)].values
        )
    ax.set_xlabel("energy fluence [eV/m\u00b2]")
    ax.set_ylabel("frequency")
    _VARS["pltFigB"].subplots_adjust(left=left_B, bottom=bottom_B, right=right_B)
    _VARS["fig_aggB"] = draw_figure_w_toolbar(
        _VARS["window"]["-CANVAS_B-"].TKCanvas,
        _VARS["pltFigB"],
        _VARS["window"]["-CANVAS_CONTROL_B-"].TKCanvas,
    )


########################################################################

_VARS["window"] = sg.Window(
    "Extensive Air Shower array visualizer", layout, finalize=True, resizable=True
)


def open_window():
    layout = [
        [
            sg.Text(
                "East is at 0, North at 90, West at 180, South at 270\nMagnetic field is set by default to Pierre Auger Observatory."
            )
        ]
    ]
    _VARS["window_help"] = sg.Window("Help", layout, modal=True)
    choice = None
    while True:
        event, values = _VARS["window_help"].read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    _VARS["window_help"].close()


def open_window_file_error():
    layout = [
        [
            sg.Text(
                "File reading failed!\nCheck whether the path is correct or whether the file format is correct.\nRequired format is CSV where first column are the stations numbers and the 2,3 and 4th column are the station coordinates.",
                font=font_emphasize,
                justification="center",
            )
        ]
    ]
    _VARS["window_read_error"] = sg.Window("ERROR", layout, modal=True)
    choice = None
    while True:
        event, values = _VARS["window_read_error"].read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    _VARS["window_read_error"].close()


drawChart(pd.DataFrame(station_dict).T, values=values)
drawChartB(pd.DataFrame(station_dict).T)

while True:
    event, values = _VARS["window"].read()
    if event == sg.WIN_CLOSED:
        break
    if event == "-CREATE_ARRAY-":
        #    print(values)
        station_dict = Create_array_layout(
            side=float(values["-STATIONS_ON_SIDE-"]),
            stations_distance_km=float(values["-STATION_DISTANCE-"]),
            array_layout=values["-ARRAY_LAYOUT-"],
            stations_layout=values["-STATIONS_LAYOUT-"],
        )
        station_DF = pd.DataFrame(station_dict).T
        updateChart(station_DF, values=values)
        updateChartB(station_DF)
        output2.update(
            "Total number of stations in the array: {}".format(len(station_dict))
        )
        fluence_info.update(fluence_def_string(station_DF))
        _VARS["window"].find_element("-CORE_X-").update(value=station_DF.x.mean())
        _VARS["window"].find_element("-CORE_Y-").update(value=station_DF.y.mean())
    if event == "-SUBMIT-":
        output.update(
            "Energy: {} [EeV] Azimuth: {} [degs] Zenith: {} [degs] Core X: {:.2f} [km] Core Y:{:.2f} [km]".format(
                values["-ENERGY-"],
                values["-AZIMUTH-"],
                values["-ZENITH-"],
                float(values["-CORE_X-"]),
                float(values["-CORE_Y-"]),
            )
        )
        azimuth = np.deg2rad(values["-AZIMUTH-"])
        zenith = np.deg2rad(values["-ZENITH-"])
        energy = values["-ENERGY-"]
        core_x_km = float(values["-CORE_X-"])
        core_y_km = float(values["-CORE_Y-"])
        station_dict = Calculate_energy_fluences(
            station_dict,
            energy,
            azimuth,
            zenith,
            core_x_km,
            core_y_km,
            B_earthVector=B_earthVector,
        )
        station_DF = pd.DataFrame(station_dict).T
        updateChart2(station_DF, values=values)
        updateChartB(station_DF)
        total_stations = len(station_DF.index)
        temp_string = fluence_def_string(station_DF)
        fluence_info.update(temp_string)
    #      print(values)
    if event == "-LOAD_CUSTOM_ARRAY-":
        #     print(values)
        try:
            station_dict = (pd.read_csv(values["-BROWSE-"], index_col=0).T).to_dict()
            read_in_is_ok = True
        except:
            read_in_is_ok = False
            open_window_file_error()
        if read_in_is_ok == True:
            station_DF = pd.DataFrame(station_dict).T
            #     print(station_DF)
            updateChart(station_DF, values=values)
            output2.update(
                "Total number of stations in the array: {}".format(len(station_dict))
            )
            fluence_info.update(fluence_def_string(station_DF))
            _VARS["window"].find_element("-CORE_X-").update(value=station_DF.x.mean())
            _VARS["window"].find_element("-CORE_Y-").update(value=station_DF.y.mean())
    if event == "-UPDATE-":
        updateChart(station_DF, values=values)
        output2.update(
            "Total number of stations in the array: {}".format(len(station_dict))
        )
        fluence_info.update(fluence_def_string(station_DF))
    if event == "-HELP-":
        open_window()
    if event == "-SET_MAGNETIC_VECTOR-":
        B_earthVector = np.array(
            [
                values["-MAG_VECTOR_X-"],
                values["-MAG_VECTOR_Y-"],
                values["-MAG_VECTOR_Z-"],
            ]
        ).astype(float)
        output_mag_field.update(
            "Magnetic field set to: {}, {}, {}.".format(*B_earthVector)
        )
