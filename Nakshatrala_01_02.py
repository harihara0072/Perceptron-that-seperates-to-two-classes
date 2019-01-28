# Nakshatrala, Hari Hara Kumar
# 1001-102-740
# 2017-09-17
# Assignment_01_02

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import tkinter as tk
class DisplayActivationFunctions:


    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.input_weight = 1
        self.input_weight1 = 1
        self.input_weight2 = 1
        #self.weight_matrix = np.array((self.input_weight1,self.input_weight2))
        self.weight_matrix = np.array((1, 1))
        self.input_matrix = np.empty([2, 4])
        self.bias = 0
        self.activation_function = "Symmetrical Hard limit"
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=5, uniform='xx')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')
        # set up the sliders
        # setting weight 1 slider
        self.input_weight_slider1 = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight1",
                                            command=lambda event: self.input_weight_slider1_callback())
        self.input_weight_slider1.set(self.weight_matrix[0])
        self.input_weight_slider1.bind("<ButtonRelease-1>", lambda event: self.input_weight_slider1_callback())
        self.input_weight_slider1.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #setting weight 2 slider

        self.input_weight_slider2 = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="Input Weight2",
                                             command=lambda event: self.input_weight_slider2_callback())
        self.input_weight_slider2.set(self.weight_matrix[1])
        self.input_weight_slider2.bind("<ButtonRelease-1>", lambda event: self.input_weight_slider2_callback())
        self.input_weight_slider2.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # setting slider for bias

        self.bias_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Bias",
                                    command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
        self.bias_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(6, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')
        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Activation Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Symmetrical Hard limit","Hyperbolic Tangent", "Linear",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetrical Hard limit")
        self.activation_function_dropdown.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # setting label for train button

        self.label_for_train_button = tk.Label(self.buttons_frame, text="Click to Train the Neuron",
                                               justify="center")
        self.label_for_train_button.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # srtting button for train

        self.Train = tk.Button(self.buttons_frame, text="Train", command=self.train_the_neuron)
        self.Train.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Setting label for Random Values

        self.label_for_Random_values = tk.Label(self.buttons_frame, text="Click to generate random Values",
                                                justify="center")
        self.label_for_Random_values.grid(row=4, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # setting button for random values

        self.Random_values = tk.Button(self.buttons_frame, text="Generate Random Values",
                                       command=self.generate_random_values)
        self.Random_values.grid(row=5, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

       # self.display_activation_function()

        print("Window size:", self.master.winfo_width(), self.master.winfo_height())

    # Function to return net value

    def net_value_callback(self,w, i, b):
        i_new = i.reshape(2, 1)
        net_value = np.dot(w, i_new) + b

        if self.activation_function == 'Symmetrical Hard limit':
            if net_value > 0:
                activation = 1.0
            else:
                activation = -1.0

        elif self.activation_function == 'Linear':
            if net_value < 1000:
                activation = np.float32(net_value[0])
            else:
                activation = 1000



        elif self.activation_function == 'Hyperbolic Tangent':
            activation = np.round((np.exp(net_value[0]) - np.exp(-net_value[0]))/(np.exp(net_value[0]) + np.exp(-net_value[0])))


        return activation

    # function to train the neuron

    def train_the_neuron(self):

        # getting weight matrix values

        #self.weight_matrix = self.get_weight_matrix()
        output = np.array((1, -1, 1, -1))
        iterations = 0

        #credits :
        #Name : Dr Noureddin Sadawi
        #Github: "https://github.com/nsadawi/perceptron/blob/master/Perceptron.java"
        #Youtube: "https://www.youtube.com/user/DrNoureddinSadawi/videos"

        for j in range(0, 50):
            globalerror = 0
            for i in range(0, 4):
                nvalue = self.net_value_callback(w=self.weight_matrix, i=self.input_matrix[:, i], b=self.bias)
                error = output[i] - nvalue
                print ("Activation is ")
                print(nvalue)
                print("error value is ")
                print(error)
                self.weight_matrix = self.weight_matrix + (error * self.input_matrix[:, i])
                self.bias = self.bias + error
                self.input_weight_slider1.set(self.weight_matrix[0])
                self.input_weight_slider2.set(self.weight_matrix[1])
                self.bias_slider.set(np.round(self.bias))
                globalerror = globalerror + (error * error)
                self.display_activation_function()
                self.decision_boundry()
                iterations = iterations + 1
            if globalerror == 0:
                print("Updated Weight are: %s", self.weight_matrix)
                print("Updated bias is: %s", self.bias)
                break
            elif j == 99:
                print("Iterations limit reached")
                break

    #function to plot the decision boundry

    def decision_boundry(self):
        self.x_intercept = np.linspace(-10, 10)
        self.y_intercept = (-self.bias - (self.x_intercept * self.weight_matrix[0])) / self.weight_matrix[1]
        self.axes.plot(self.x_intercept, self.y_intercept)
        self.axes.fill_between(self.x_intercept, self.y_intercept, 10, interpolate='True', color='#FFA07A', alpha=0.5)
        self.axes.fill_between(self.x_intercept, -10, self.y_intercept, interpolate='True',  color='#90EE90', alpha=0.5)
        self.axes.xaxis.set_visible(True)
        self.canvas.draw()

    #function to generate random input values

    def generate_random_values(self):
        #self.input_weight1 = 1
        #self.input_weight2 = 1
        self.bias = 0
        self.weight_matrix = np.array((1, 1))
        self.input_weight_slider1.set(self.weight_matrix[0])
        self.input_weight_slider2.set(self.weight_matrix[0])
        self.bias = 0
        self.bias_slider.set(self.bias)
        self.inputs1 = np.random.randint(-10, 10, size=4)
        self.inputs2 = np.random.randint(-10, 10, size=4)
        self.input_matrix = np.vstack((self.inputs1, self.inputs2))
        print("Input Matrix Generated is:")
        print(self.input_matrix)
        self.display_activation_function()

    def display_activation_function(self):

        self.axes.cla()
        self.axes.cla()
        #colors = ["#006400", "#FF0000"]
        #shapes = ['o', '*', 'o', '*']
        #x = self.input_matrix[0]
        #y = self.input_matrix[1]
        #self.axes.scatter(x, y, color=colors)
        self.axes.scatter(self.input_matrix[0,0], self.input_matrix[1,0], marker='*', color="#00008B")
        self.axes.scatter(self.input_matrix[0, 1], self.input_matrix[1, 1], marker='o', color="#00008B")
        self.axes.scatter(self.input_matrix[0, 2], self.input_matrix[1, 2], marker='*', color="#00008B")
        self.axes.scatter(self.input_matrix[0, 3], self.input_matrix[1, 3], marker='o', color="#00008B")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.axes.xaxis.set_visible(True)
        plt.title(self.activation_function)
        self.canvas.draw()

    def input_weight_slider1_callback(self):
        self.weight_matrix[0] = self.input_weight_slider1.get()
        self.display_activation_function()
        self.decision_boundry()

    def input_weight_slider2_callback(self):
        self.weight_matrix[1] = self.input_weight_slider2.get()
        self.display_activation_function()
        self.decision_boundry()

    def bias_slider_callback(self):
        self.bias = self.bias_slider.get()
        self.display_activation_function()
        self.decision_boundry()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        self.display_activation_function()