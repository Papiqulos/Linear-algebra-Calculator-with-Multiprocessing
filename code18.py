from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import multiprocessing as mp
import time
from tkinter import messagebox as mb


class RandomMatrix:
    """This class contains methods that create either 1 or 2 random matrices
    with given dimensions whose elements are integers between 0 and 10"""
    @staticmethod
    def random_matrix(a, b=0):
        """Creates a random matrix with dimensions a x b"""
        matrix = np.random.randint(10, size=(a, b))
        return matrix

    @staticmethod
    def two_random_matrices(a, b=0, c=0):
        """Creates a random matrix with dimensions a x b and another one with
        dimensions b x c"""
        matrix_A = np.random.randint(10, size=(a, b))
        matrix_B = np.random.randint(10, size=(b, c))
        return matrix_A, matrix_B


class SimpleCalculation:
    """This class contains methods that perform basic linear algebra
    calculations (without the help of multiprocessing)"""
    @staticmethod
    def matrix_add(matrix_A, matrix_B):
        return matrix_A + matrix_B

    @staticmethod
    def matrix_sub(matrix_A, matrix_B):
        return matrix_A - matrix_B

    @staticmethod
    def matrix_mul_num(matrix, number):
        return matrix * float(number)

    @staticmethod
    def matrix_mul(matrix_A, matrix_B):
        return np.matmul(matrix_A, matrix_B)

    @staticmethod
    def matrix_power(matrix, power):
        return np.linalg.matrix_power(matrix, power)

    @staticmethod
    def matrix_det(matrix):
        return np.linalg.det(matrix)

    @staticmethod
    def matrix_inv(matrix):
        return np.linalg.inv(matrix)

    @staticmethod
    def matrix_trans(matrix):
        return np.transpose(matrix)

    @staticmethod
    def matrix_rank(matrix):
        return np.linalg.matrix_rank(matrix)

    @staticmethod
    def matrix_trace(matrix):
        return np.trace(matrix)


class MultiprocessingCalculation:
    """This class contains methods that calculate the product of two matrices
    as well as the matrix raised to some power using multiple simultaneous
    processes that decrease the computation time"""
    @staticmethod
    def block_shaped(arr):
        """Divides a square array into 4 equal arrays"""
        nrc = int(arr.shape[0] / 2)
        return arr.reshape(2, nrc, -1, nrc).swapaxes(1, 2).reshape(-1, nrc, nrc)

    @staticmethod
    def simple_mul(arrA, arrB, pos, return_dict):
        """Calculates the product of two arrays and inserts it into a dictionary"""
        rtrn = np.matmul(arrA, arrB)
        return_dict[pos] = rtrn

    @staticmethod
    def array_sum(return_dict):
        """Calculates the sum of two products of arrays"""
        c11 = dict(return_dict)['11'] + dict(return_dict)['23']
        c12 = dict(return_dict)['12'] + dict(return_dict)['24']
        c21 = dict(return_dict)['31'] + dict(return_dict)['43']
        c22 = dict(return_dict)['32'] + dict(return_dict)['44']
        return c11, c12, c21, c22

    @staticmethod
    def connect(c11, c12, c21, c22):
        """Concatenates the four sub-arrays into the final array"""
        above = np.concatenate((c11, c12), axis=1)
        below = np.concatenate((c21, c22), axis=1)
        final = np.concatenate((above, below), axis=0)
        return final

    @staticmethod
    def zero_pad(arr):
        """Appends a row and a column of zeros into the array"""
        dim = arr.shape[0]
        zero1 = np.zeros((dim, 1), dtype='int32')
        arr = np.concatenate((arr, zero1), axis=1)
        zero2 = np.zeros((1, dim + 1), dtype='int32')
        arr = np.concatenate((arr, zero2), axis=0)
        return arr

    @staticmethod
    def zero_pad_row(arr):
        """Appends a row of zeros into the array"""
        columns = arr.shape[1]
        zero = np.zeros((1, columns), dtype='int32')
        return np.concatenate((arr, zero), axis=0)

    @staticmethod
    def zero_pad_col(arr):
        """Appends a column of zeros into the array"""
        rows = arr.shape[0]
        zero = np.zeros((rows, 1), dtype='int32')
        return np.concatenate((arr, zero), axis=1)

    @staticmethod
    def a_split_mul(arrA, arrB):
        """Splits array A into two equal arrays and performs the multiplication
        between the two equal arrays and array B"""
        a1, a2 = np.vsplit(arrA, 2)

        manager = mp.Manager()
        return_dict = manager.dict()

        pool = mp.Pool()
        pool.starmap(MultiprocessingCalculation.simple_mul, [(a1, arrB, '1', return_dict),
                                                             (a2, arrB, '2', return_dict), ])
        pool.close()
        pool.join()

        return np.concatenate((dict(return_dict)['1'], dict(return_dict)['2']), axis=0)

    @staticmethod
    def b_split_mul(arrA, arrB):
        """Splits array B into two equal arrays and performs the multiplication
        between the two equal arrays and array A"""
        b1, b2 = np.hsplit(arrB, 2)

        manager = mp.Manager()
        return_dict = manager.dict()

        pool = mp.Pool()
        pool.starmap(MultiprocessingCalculation.simple_mul, [(arrA, b1, '1', return_dict),
                                                             (arrA, b2, '2', return_dict), ])
        pool.close()
        pool.join()

        return np.concatenate((dict(return_dict)['1'], dict(return_dict)['2']), axis=1)

    @staticmethod
    def both_split_mul(arrA, arrB):
        """Splits both arrays into four equal arrays and performs the
        multiplication between the four equal arrays"""
        a1, a2 = np.hsplit(arrA, 2)
        b1, b2 = np.vsplit(arrB, 2)

        manager = mp.Manager()
        return_dict = manager.dict()

        pool = mp.Pool()
        pool.starmap(MultiprocessingCalculation.simple_mul, [(a1, b1, '1', return_dict), (a2, b2, '2', return_dict), ])
        pool.close()
        pool.join()

        return dict(return_dict)['1'] + dict(return_dict)['2']

    @staticmethod
    def square_mul(arrA, arrB):
        """Splits both arrays into eight equal arrays and performs the
        multiplication between the eight equal arrays"""
        a11, a12, a21, a22 = MultiprocessingCalculation.block_shaped(arrA)
        b11, b12, b21, b22 = MultiprocessingCalculation.block_shaped(arrB)

        manager = mp.Manager()
        return_dict = manager.dict()

        pool = mp.Pool()
        pool.starmap(MultiprocessingCalculation.simple_mul,
                     [(a11, b11, '11', return_dict), (a12, b21, '23', return_dict),
                      (a11, b12, '12', return_dict), (a12, b22, '24', return_dict),
                      (a21, b11, '31', return_dict), (a22, b21, '43', return_dict),
                      (a21, b12, '32', return_dict), (a22, b22, '44', return_dict), ])
        pool.close()
        pool.join()

        c11, c12, c21, c22 = MultiprocessingCalculation.array_sum(return_dict)
        result = MultiprocessingCalculation.connect(c11, c12, c21, c22)

        return result

    @staticmethod
    def multiplication(arrA, arrB):
        """Carries out various checks and calls the according methods"""
        a = arrA.shape[0]
        b = arrA.shape[1]
        c = arrB.shape[1]

        if a == b == c:  # Square matrix
            if a % 2 != 0:  # Check if Matrix's dimensions are odd numbers
                arrA = MultiprocessingCalculation.zero_pad(arrA)
                arrB = MultiprocessingCalculation.zero_pad(arrB)
                result = MultiprocessingCalculation.square_mul(arrA, arrB)
                result = np.delete(result, -1, 0)
                result = np.delete(result, -1, 1)
                return result
            else:
                result = MultiprocessingCalculation.square_mul(arrA, arrB)
                return result

        elif max(a, b, c) == a:  # Matrix A's rows is the largest dimension
            if a % 2 != 0:  # Check if Matrix A's rows is an odd number
                arrA = MultiprocessingCalculation.zero_pad_row(arrA)
                result = MultiprocessingCalculation.a_split_mul(arrA, arrB)
                result = np.delete(result, -1, 0)
                return result
            else:
                result = MultiprocessingCalculation.a_split_mul(arrA, arrB)
                return result

        elif max(a, b, c) == c:  # Matrix B's columns is the largest dimension
            if c % 2 != 0:  # Check if Matrix B's columns is an odd number
                arrB = MultiprocessingCalculation.zero_pad_col(arrB)
                result = MultiprocessingCalculation.b_split_mul(arrA, arrB)
                result = np.delete(result, -1, 1)
                return result
            else:
                result = MultiprocessingCalculation.b_split_mul(arrA, arrB)
                return result

        elif max(a, b, c) == b:  # Matrix A's columns (and Matrix B's rows) is the largest dimension
            if b % 2 != 0:  # Check if Matrix A's columns (and Matrix B's rows) is an odd number
                arrA = MultiprocessingCalculation.zero_pad_col(arrA)
                arrB = MultiprocessingCalculation.zero_pad_row(arrB)
            if a % 2 != 0:  # Check if Matrix A's rows is an odd number
                arrA = MultiprocessingCalculation.zero_pad_row(arrA)
            if c % 2 != 0:  # Check if Matrix B's columns is an odd number
                arrB = MultiprocessingCalculation.zero_pad_col(arrB)

            result = MultiprocessingCalculation.both_split_mul(arrA, arrB)

            rows, columns = result.shape
            if rows == a + 1:
                result = np.delete(result, -1, 0)
            if columns == c + 1:
                result = np.delete(result, -1, 1)

            return result

    @staticmethod
    def matrix_power(matrix, power):
        """Calculates the matrix raised to some positive integer"""
        if power % 2 == 0:  # Power is an even number
            matrix_square = MultiprocessingCalculation.multiplication(matrix, matrix)
            matrix2 = matrix_square
            for i in range(0, power - 1, 2):
                matrix2 = MultiprocessingCalculation.multiplication(matrix_square, matrix2)
            return matrix2

        elif power % 2 != 0:  # Power is an odd number
            matrix_square = MultiprocessingCalculation.multiplication(matrix, matrix)
            matrix2 = matrix_square
            for i in range(0, power - 2, 2):
                matrix2 = MultiprocessingCalculation.multiplication(matrix_square, matrix2)
            return MultiprocessingCalculation.multiplication(matrix, matrix2)


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.iconbitmap('matrix_ico.ico')
        self.color_bg1 = '#293241'      #
        self.color_bg2 = '#3d5a80'      #
        self.color_button1 = '#3d5a80'  #
        self.color_button2 = '#d8bc66'  # Color palette
        self.color_text1 = '#ee6c4d'    #
        self.color_text2 = '#98c1d9'    #
        self.color_text3 = '#ffffff'    #
        self.root.geometry('1200x800')
        self.root.resizable(False, False)
        self.root.title('Matrix Calculator')
        self.dim_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.f_left = Frame(self.root, bg=self.color_bg1)  # Frame containing button frame and logo
        self.f_left.pack(side='left', fill='y')
        self.logo = Canvas(self.f_left, bg=self.color_bg1, highlightbackground=self.color_bg1, width=212, height=150)  # Canvas containing upper left logo
        self.logo.grid(row=0, column=0, sticky='N', pady=20)
        self.f_buttons = Frame(self.f_left, bg=self.color_bg2, padx=10, pady=10)  # Frame containing function buttons
        self.f_buttons.grid(row=1, column=0, pady=30)
        global photo
        photo = Image.open("logo.png").resize((150, 117), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(photo)
        self.logo.create_image(40, 15, image=photo, anchor=NW)

        # Creating function buttons
        self.b_add_sub = Button(self.f_buttons, text='Matrix\nAddition/Subtraction', font=('Arial', 13),
                                bg=self.color_button1, fg=self.color_text2, activebackground=self.color_button2,
                                activeforeground=self.color_text2, pady=5, command=lambda: GUI.add_sub(self))
        self.b_mul_num = Button(self.f_buttons, text='Matrix Multiplication\nby Number', font=('Arial', 13),
                                bg=self.color_button1, fg=self.color_text2, activebackground=self.color_button2,
                                activeforeground=self.color_text2, pady=5, command=lambda: GUI.mul_num(self))
        self.b_mul = Button(self.f_buttons, text='Matrix Multiplication', font=('Arial', 13), bg=self.color_button1,
                            fg=self.color_text2, activebackground=self.color_button2, activeforeground=self.color_text2,
                            pady=5, command=lambda: GUI.mul(self))
        self.b_power = Button(self.f_buttons, text='Matrix Power', font=('Arial', 13), bg=self.color_button1,
                              fg=self.color_text2, activebackground=self.color_button2,
                              activeforeground=self.color_text2, pady=5, command=lambda: GUI.power(self))
        self.b_det = Button(self.f_buttons, text='Matrix\nDeterminant', font=('Arial', 13), bg=self.color_button1,
                            fg=self.color_text2, activebackground=self.color_button2, activeforeground=self.color_text2,
                            pady=5, command=lambda: GUI.det(self))
        self.b_inv = Button(self.f_buttons, text='Inverse Matrix', font=('Arial', 13), bg=self.color_button1,
                            fg=self.color_text2, activebackground=self.color_button2, activeforeground=self.color_text2,
                            pady=5, command=lambda: GUI.inv(self))
        self.b_trans = Button(self.f_buttons, text='Matrix\nTranspose', font=('Arial', 13), bg=self.color_button1,
                              fg=self.color_text2, activebackground=self.color_button2,
                              activeforeground=self.color_text2, pady=5, command=lambda: GUI.trans(self))
        self.b_rank = Button(self.f_buttons, text='Matrix Rank', font=('Arial', 13), bg=self.color_button1,
                             fg=self.color_text2, activebackground=self.color_button2,
                             activeforeground=self.color_text2, pady=5, command=lambda: GUI.rank(self))
        self.b_trace = Button(self.f_buttons, text='Matrix Trace', font=('Arial', 13), bg=self.color_button1,
                              fg=self.color_text2, activebackground=self.color_button2,
                              activeforeground=self.color_text2, pady=5, command=lambda: GUI.trace(self))

        # Packing function buttons
        self.b_add_sub.pack(fill='x')
        self.b_mul_num.pack(fill='x')
        self.b_mul.pack(fill='x')
        self.b_power.pack(fill='x')
        self.b_det.pack(fill='x')
        self.b_inv.pack(fill='x')
        self.b_trans.pack(fill='x')
        self.b_rank.pack(fill='x')
        self.b_trace.pack(fill='x')

        self.f_main = Frame(self.root, bg=self.color_bg1)  # Frame containing func descriptions and dimensions selection
        self.f_main.pack(side='left', expand=True, fill='both')

        self.title = Label(self.f_main, text='Matrix-18 Calculator', font=('Arial', 50, 'bold'),
                           bg=self.color_bg1, fg=self.color_text1)  # Main title
        self.title.pack()
        self.info = Label(self.f_main, text='''
Matrix-18 is a convenient and easy to use application for 
the calculation of basic matrix operations between matrices 
of various dimensions giving results with speed and accuracy.
With the help of this calculator you can: add/ subtract matrices, 
multiply matrices by a number, multiply matrices,  find a matrix 
determinant, rank, trace and calculate the inverse and the 
transpose matrix''', font=('Arial', 20), bg=self.color_bg1,
                          fg=self.color_text1)  # Main app description
        self.info.pack()

    """Each method below displays the correct widgets on the main (left)
    frame for entering the dimensions of the matrix (or matrices),
    according to the desired calculation"""
    def add_sub(self):
        GUI.clear_frame(self)

        self.title = Label(self.f_main, text='Matrix Addition and Subtraction Calculator', font=('Arial', 30, 'bold'),
                           bg=self.color_bg1, fg=self.color_text1)
        self.title.pack(pady=(30, 0))

        self.desc = Label(self.f_main, text='''Matrix addition/subtraction is the operation of adding/subtracting 
two matrices by adding/subtracting the corresponding elements together.''', font=('Arial', 15),
                          bg=self.color_bg1, fg=self.color_text1)
        self.desc.pack(pady=(30, 0))

        self.f_dims = LabelFrame(self.f_main, text='Custom Matrices', padx=100, pady=10, bg=self.color_bg1,
                                 fg=self.color_text3, relief=GROOVE)
        self.f_dims.pack(pady=(200, 50))

        self.dim_text = Label(self.f_dims, text='Matrices dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.dim_text.grid(row=0, column=0)

        self.dim_m = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim_m.current(0)
        self.dim_m.grid(row=0, column=1)

        self.X_text = Label(self.f_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.X_text.grid(row=0, column=2)

        self.dim_n = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim_n.current(0)
        self.dim_n.grid(row=0, column=3)

        self.op_text = Label(self.f_dims, text='Operation', bg=self.color_bg1, fg=self.color_text2)
        self.op_text.grid(row=0, column=4, padx=(30, 2))

        self.op = ttk.Combobox(self.f_dims, values=['+', '-'], width=2, state='readonly')
        self.op.current(0)
        self.op.grid(row=0, column=5)

        self.set = Button(self.f_dims, text='Set matrices', bg=self.color_bg1, fg=self.color_text2,
                          activebackground=self.color_bg1,
                          activeforeground=self.color_text2, relief=RIDGE,
                          command=lambda: GUI.set_matrix(self, 'add_sub', int(self.dim_m.get()), int(self.dim_n.get())))
        self.set.grid(row=0, column=6, padx=(100, 0))

        self.f_rand_dims = LabelFrame(self.f_main, text='Random Matrices', padx=100, pady=10, bg=self.color_bg1,
                                      fg=self.color_text3, relief=GROOVE)
        self.f_rand_dims.pack()

        self.rand_dim_text = Label(self.f_rand_dims, text='Matrices dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.rand_dim_text.grid(row=0, column=0)

        self.rand_dim_m = Entry(self.f_rand_dims, width=6)
        self.rand_dim_m.delete(0, END)
        self.rand_dim_m.insert(0, '2')
        self.rand_dim_m.grid(row=0, column=1)

        self.rand_X_text = Label(self.f_rand_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.rand_X_text.grid(row=0, column=2)

        self.rand_dim_n = Entry(self.f_rand_dims, width=6)
        self.rand_dim_n.delete(0, END)
        self.rand_dim_n.insert(0, '2')
        self.rand_dim_n.grid(row=0, column=3)

        self.rand_op_text = Label(self.f_rand_dims, text='Operation', bg=self.color_bg1, fg=self.color_text2)
        self.rand_op_text.grid(row=0, column=4, padx=(30, 2))

        self.rand_op = ttk.Combobox(self.f_rand_dims, values=['+', '-'], width=2)
        self.rand_op.current(0)
        self.rand_op.grid(row=0, column=5)

        self.rand_set = Button(self.f_rand_dims, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=9,
                               activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                               command=lambda: GUI.rand_calculate(self, 'add_sub', self.rand_dim_m.get(),
                                                                  self.rand_dim_n.get()))
        self.rand_set.grid(row=0, column=6, padx=(100, 0))

    def mul_num(self):
        GUI.clear_frame(self)

        self.title = Label(self.f_main, text='Matrix Multiplication By Number Calculator', font=('Arial', 30, 'bold'),
                           bg=self.color_bg1, fg=self.color_text1)
        self.title.pack(pady=(30, 0))

        self.desc = Label(self.f_main, text='''Matrix Multiplication By Number is the operation of multiplying
every element of the matrix by a certain number.''', font=('Arial', 15), bg=self.color_bg1, fg=self.color_text1)
        self.desc.pack(pady=(30, 0))

        self.f_dims = LabelFrame(self.f_main, text='Custom Matrix', padx=100, pady=10, bg=self.color_bg1,
                                 fg=self.color_text3, relief=GROOVE)
        self.f_dims.pack(pady=(200, 50))

        self.dim_text = Label(self.f_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.dim_text.grid(row=0, column=0)

        self.dim_m = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim_m.current(0)
        self.dim_m.grid(row=0, column=1)

        self.X_text = Label(self.f_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.X_text.grid(row=0, column=2)

        self.dim_n = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim_n.current(0)
        self.dim_n.grid(row=0, column=3)

        self.num_text = Label(self.f_dims, text='Multiply by', bg=self.color_bg1, fg=self.color_text2)
        self.num_text.grid(row=0, column=4, padx=(30, 2))

        self.num_entry = Entry(self.f_dims, width=3)
        self.num_entry.insert(0, '1')
        self.num_entry.grid(row=0, column=5)

        self.set = Button(self.f_dims, text='Set matrix', bg=self.color_bg1, fg=self.color_text2,
                          activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                          command=lambda: GUI.set_matrix(self, 'mul_num', int(self.dim_m.get()), int(self.dim_n.get())))
        self.set.grid(row=0, column=6, padx=(100, 0))

        self.f_rand_dims = LabelFrame(self.f_main, text='Random Matrix', padx=100, pady=10, bg=self.color_bg1,
                                      fg=self.color_text3, relief=GROOVE)
        self.f_rand_dims.pack()

        self.rand_dim_text = Label(self.f_rand_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.rand_dim_text.grid(row=0, column=0)

        self.rand_dim_m = Entry(self.f_rand_dims, width=6)
        self.rand_dim_m.delete(0, END)
        self.rand_dim_m.insert(0, '2')
        self.rand_dim_m.grid(row=0, column=1)

        self.rand_X_text = Label(self.f_rand_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.rand_X_text.grid(row=0, column=2)

        self.rand_dim_n = Entry(self.f_rand_dims, width=6)
        self.rand_dim_n.delete(0, END)
        self.rand_dim_n.insert(0, '2')
        self.rand_dim_n.grid(row=0, column=3)

        self.rand_num_text = Label(self.f_rand_dims, text='Multiply by', bg=self.color_bg1, fg=self.color_text2)
        self.rand_num_text.grid(row=0, column=4, padx=(30, 2))

        self.rand_num_entry = Entry(self.f_rand_dims, width=3)
        self.rand_num_entry.insert(0, '1')
        self.rand_num_entry.grid(row=0, column=5)

        self.rand_set = Button(self.f_rand_dims, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=3,
                               activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                               command=lambda: GUI.rand_calculate(self, 'mul_num', self.rand_dim_m.get(),
                                                                  self.rand_dim_n.get()))
        self.rand_set.grid(row=0, column=6, padx=(100, 0))

    def mul(self):
        GUI.clear_frame(self)

        self.title = Label(self.f_main, text='Matrix Multiplication Calculator', font=('Arial', 30, 'bold'),
                           bg=self.color_bg1, fg=self.color_text1)
        self.title.pack(pady=(30, 0))

        self.desc = Label(self.f_main, text='''Matrix Multiplication is a binary operation that produces 
a matrix from two matrices. For matrix multiplication, 
the number of columns in the first matrix must be equal 
to the number of rows in the second matrix. The resulting matrix, 
known as the matrix product, has the number of rows 
of the first and the number of columns of the second matrix.''', font=('Arial', 15), bg=self.color_bg1,
                          fg=self.color_text1)
        self.desc.pack(pady=(30, 0))

        self.f_dims = LabelFrame(self.f_main, text='Custom Matrices', padx=100, pady=10, bg=self.color_bg1,
                                 fg=self.color_text3, relief=GROOVE)
        self.f_dims.pack(pady=(200, 50))

        self.dimA_text = Label(self.f_dims, text='Matrix A dimensions:', bg=self.color_bg1, fg=self.color_text2)
        self.dimA_text.grid(row=0, column=0, pady=10)
        self.dimB_text = Label(self.f_dims, text='Matrix B dimensions:', bg=self.color_bg1, fg=self.color_text2)
        self.dimB_text.grid(row=1, column=0)

        def callback(iv):
            iv.set(iv.get())

        iv1 = IntVar()
        iv1.trace("w", lambda name, index, mode, iv1=iv1: callback(iv1))

        self.dimA_m = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dimA_m.current(0)
        self.dimA_m.grid(row=0, column=1)

        self.X_text = Label(self.f_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.X_text.grid(row=0, column=2)

        self.dimA_n = ttk.Combobox(self.f_dims, textvariable=iv1, values=self.dim_values, width=3, state='readonly')
        self.dimA_n.current(0)
        self.dimA_n.grid(row=0, column=3)

        self.dimB_m = ttk.Combobox(self.f_dims, textvariable=iv1, values=self.dim_values, width=3, state='readonly')
        self.dimB_m.current(0)
        self.dimB_m.grid(row=1, column=1)

        self.X_text = Label(self.f_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.X_text.grid(row=1, column=2)

        self.dimB_n = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dimB_n.current(0)
        self.dimB_n.grid(row=1, column=3)

        self.set = Button(self.f_dims, text='Set matrices', bg=self.color_bg1, fg=self.color_text2,
                          activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                          command=lambda: GUI.set_matrix(self, 'mul', int(self.dimA_m.get()), int(self.dimA_n.get()),
                                                         int(self.dimB_n.get())))
        self.set.grid(row=0, column=6, rowspan=2, padx=(100, 0))

        self.f_rand_dims = LabelFrame(self.f_main, text='Random Matrices', padx=100, pady=10, bg=self.color_bg1,
                                      fg=self.color_text3, relief=GROOVE)
        self.f_rand_dims.pack()

        self.rand_dimA_text = Label(self.f_rand_dims, text='Matrix A dimensions:', bg=self.color_bg1,
                                    fg=self.color_text2)
        self.rand_dimA_text.grid(row=0, column=0, pady=10)
        self.rand_dimB_text = Label(self.f_rand_dims, text='Matrix B dimensions:', bg=self.color_bg1,
                                    fg=self.color_text2)
        self.rand_dimB_text.grid(row=1, column=0)

        iv2 = StringVar()
        iv2.set('2')
        iv2.trace("w", lambda name, index, mode, iv2=iv2: callback(iv2))

        self.rand_dimA_m = Entry(self.f_rand_dims, width=6)
        self.rand_dimA_m.insert(0, '2')
        self.rand_dimA_m.grid(row=0, column=1)

        self.rand_X_text = Label(self.f_rand_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.rand_X_text.grid(row=0, column=2)

        self.rand_dimA_n = Entry(self.f_rand_dims, width=6, textvariable=iv2)
        self.rand_dimA_n.grid(row=0, column=3)

        self.rand_dimB_m = Entry(self.f_rand_dims, textvariable=iv2, width=6)
        self.rand_dimB_m.grid(row=1, column=1)

        self.rand_X_text = Label(self.f_rand_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.rand_X_text.grid(row=1, column=2)

        self.rand_dimB_n = Entry(self.f_rand_dims, width=6)
        self.rand_dimB_n.insert(0, '2')
        self.rand_dimB_n.grid(row=1, column=3)

        self.rand_set = Button(self.f_rand_dims, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=9,
                               activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                               command=lambda: GUI.rand_calculate(self, 'mul', self.rand_dimA_m.get(),
                                                                  self.rand_dimA_n.get(),
                                                                  self.rand_dimB_n.get()))
        self.rand_set.grid(row=0, column=6, rowspan=2, padx=(100, 0))

    def power(self):
        GUI.clear_frame(self)

        self.title = Label(self.f_main, text='Matrix Power Calculator', font=('Arial', 30, 'bold'),
                           bg=self.color_bg1, fg=self.color_text1)
        self.title.pack(pady=(30, 0))

        self.desc = Label(self.f_main, text='''Matrix power is obtained by multiplication matrix by itself 'n' times.
The matrix must be square in order to raise it to a power.''', font=('Arial', 15), bg=self.color_bg1,
                          fg=self.color_text1)
        self.desc.pack(pady=(30, 0))

        self.f_dims = LabelFrame(self.f_main, text='Custom Matrix', padx=100, pady=10, bg=self.color_bg1,
                                 fg=self.color_text3, relief=GROOVE)
        self.f_dims.pack(pady=(200, 50))

        self.dim_text = Label(self.f_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.dim_text.grid(row=0, column=0)

        self.dim = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim.current(0)
        self.dim.grid(row=0, column=1)

        self.power_text = Label(self.f_dims, text='Power', bg=self.color_bg1, fg=self.color_text2)
        self.power_text.grid(row=0, column=4, padx=(30, 2))

        self.power_entry = Entry(self.f_dims, width=3)
        self.power_entry.insert(0, '1')
        self.power_entry.grid(row=0, column=5)

        self.set = Button(self.f_dims, text='Set matrix', bg=self.color_bg1, fg=self.color_text2,
                          activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                          command=lambda: GUI.set_matrix(self, 'power', int(self.dim.get())))
        self.set.grid(row=0, column=6, padx=(100, 0))

        self.f_rand_dims = LabelFrame(self.f_main, text='Random Matrix', padx=100, pady=10, bg=self.color_bg1,
                                      fg=self.color_text3, relief=GROOVE)
        self.f_rand_dims.pack()

        self.rand_dim_text = Label(self.f_rand_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.rand_dim_text.grid(row=0, column=0)

        self.rand_dim = Entry(self.f_rand_dims, width=6)
        self.rand_dim.delete(0, END)
        self.rand_dim.insert(0, '2')
        self.rand_dim.grid(row=0, column=1)

        self.rand_power_text = Label(self.f_rand_dims, text='Power', bg=self.color_bg1, fg=self.color_text2)
        self.rand_power_text.grid(row=0, column=4, padx=(30, 2))

        self.rand_power_entry = Entry(self.f_rand_dims, width=3)
        self.rand_power_entry.insert(0, '1')
        self.rand_power_entry.grid(row=0, column=5)

        self.rand_set = Button(self.f_rand_dims, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=3,
                               activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                               command=lambda: GUI.rand_calculate(self, 'power', self.rand_dim.get()))
        self.rand_set.grid(row=0, column=6, padx=(100, 0))

    def det(self):
        GUI.clear_frame(self)

        self.title = Label(self.f_main, text='Determinant Calculator', font=('Arial', 30, 'bold'),
                           bg=self.color_bg1, fg=self.color_text1)
        self.title.pack(pady=(30, 0))

        self.desc = Label(self.f_main, text='''the determinant is a scalar value 
that can be computed from the elements 
of a square matrix and encodes certain properties 
of the linear transformation described by the matrix.
The determinant of a matrix A is denoted det(A), det A, or |A|.''', font=('Arial', 15), bg=self.color_bg1,
                          fg=self.color_text1)
        self.desc.pack(pady=(30, 0))

        self.f_dims = LabelFrame(self.f_main, text='Custom Matrix', padx=100, pady=10, bg=self.color_bg1,
                                 fg=self.color_text3, relief=GROOVE)
        self.f_dims.pack(pady=(200, 50))

        self.dim_text = Label(self.f_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.dim_text.grid(row=0, column=0)

        self.dim = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim.current(0)
        self.dim.grid(row=0, column=1)

        self.set = Button(self.f_dims, text='Set matrix', bg=self.color_bg1, fg=self.color_text2,
                          activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                          command=lambda: GUI.set_matrix(self, 'det', int(self.dim.get())))
        self.set.grid(row=0, column=6, padx=(100, 0))

        self.f_rand_dims = LabelFrame(self.f_main, text='Random Matrix', padx=100, pady=10, bg=self.color_bg1,
                                      fg=self.color_text3, relief=GROOVE)
        self.f_rand_dims.pack()

        self.rand_dim_text = Label(self.f_rand_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.rand_dim_text.grid(row=0, column=0)

        self.rand_dim = Entry(self.f_rand_dims, width=6)
        self.rand_dim.insert(0, '2')
        self.rand_dim.grid(row=0, column=1)

        self.rand_set = Button(self.f_rand_dims, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=3,
                               activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                               command=lambda: GUI.rand_calculate(self, 'det', self.rand_dim.get()))
        self.rand_set.grid(row=0, column=6, padx=(100, 0))

    def inv(self):
        GUI.clear_frame(self)

        self.title = Label(self.f_main, text='Inverse Matrix Calculator', font=('Arial', 30, 'bold'),
                           bg=self.color_bg1, fg=self.color_text1)
        self.title.pack(pady=(30, 0))

        self.desc = Label(self.f_main, text='''In linear algebra, an n-by-n square matrix A 
is called invertible (also nonsingular or nondegenerate), 
if there exists an n-by-n square matrix B such that AB=BA=I''', font=('Arial', 15), bg=self.color_bg1,
                          fg=self.color_text1)
        self.desc.pack(pady=(30, 0))

        self.f_dims = LabelFrame(self.f_main, text='Custom Matrix', padx=100, pady=10, bg=self.color_bg1,
                                 fg=self.color_text3, relief=GROOVE)
        self.f_dims.pack(pady=(200, 50))

        self.dim_text = Label(self.f_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.dim_text.grid(row=0, column=0)

        self.dim = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim.current(0)
        self.dim.grid(row=0, column=1)

        self.set = Button(self.f_dims, text='Set matrix', bg=self.color_bg1, fg=self.color_text2,
                          activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                          command=lambda: GUI.set_matrix(self, 'inv', int(self.dim.get())))
        self.set.grid(row=0, column=6, padx=(100, 0))

        self.f_rand_dims = LabelFrame(self.f_main, text='Random Matrix', padx=100, pady=10, bg=self.color_bg1,
                                      fg=self.color_text3, relief=GROOVE)
        self.f_rand_dims.pack()

        self.rand_dim_text = Label(self.f_rand_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.rand_dim_text.grid(row=0, column=0)

        self.rand_dim = Entry(self.f_rand_dims, width=6)
        self.rand_dim.insert(0, '2')
        self.rand_dim.grid(row=0, column=1)

        self.rand_set = Button(self.f_rand_dims, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=3,
                               activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                               command=lambda: GUI.rand_calculate(self, 'inv', self.rand_dim.get()))
        self.rand_set.grid(row=0, column=6, padx=(100, 0))

    def trans(self):
        GUI.clear_frame(self)

        self.title = Label(self.f_main, text='Matrix Transpose Calculator', font=('Arial', 30, 'bold'),
                           bg=self.color_bg1, fg=self.color_text1)
        self.title.pack(pady=(30, 0))

        self.desc = Label(self.f_main, text='''the transpose of a matrix is an operator 
which flips a matrix over its diagonal; 
that is, it switches the row and 
column indices of the matrix A 
by producing another matrix, 
often denoted by AT (among other notations).''', font=('Arial', 15), bg=self.color_bg1, fg=self.color_text1)
        self.desc.pack(pady=(30, 0))

        self.f_dims = LabelFrame(self.f_main, text='Custom Matrix', padx=100, pady=10, bg=self.color_bg1,
                                 fg=self.color_text3, relief=GROOVE)
        self.f_dims.pack(pady=(200, 50))

        self.dim_text = Label(self.f_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.dim_text.grid(row=0, column=0)

        self.dim_m = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim_m.current(0)
        self.dim_m.grid(row=0, column=1)

        self.X_text = Label(self.f_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.X_text.grid(row=0, column=2)

        self.dim_n = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim_n.current(0)
        self.dim_n.grid(row=0, column=3)

        self.set = Button(self.f_dims, text='Set matrix', bg=self.color_bg1, fg=self.color_text2,
                          activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                          command=lambda: GUI.set_matrix(self, 'trans', int(self.dim_m.get()), int(self.dim_n.get())))
        self.set.grid(row=0, column=6, padx=(100, 0))

        self.f_rand_dims = LabelFrame(self.f_main, text='Random Matrix', padx=100, pady=10, bg=self.color_bg1,
                                      fg=self.color_text3, relief=GROOVE)
        self.f_rand_dims.pack()

        self.rand_dim_text = Label(self.f_rand_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.rand_dim_text.grid(row=0, column=0)

        self.rand_dim_m = Entry(self.f_rand_dims, width=6)
        self.rand_dim_m.insert(0, '2')
        self.rand_dim_m.grid(row=0, column=1)

        self.rand_X_text = Label(self.f_rand_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.rand_X_text.grid(row=0, column=2)

        self.rand_dim_n = Entry(self.f_rand_dims, width=6)
        self.rand_dim_n.insert(0, '2')
        self.rand_dim_n.grid(row=0, column=3)

        self.rand_set = Button(self.f_rand_dims, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=3,
                               activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                               command=lambda: GUI.rand_calculate(self, 'trans', self.rand_dim_m.get(),
                                                                  self.rand_dim_n.get()))
        self.rand_set.grid(row=0, column=6, padx=(100, 0))

    def rank(self):
        GUI.clear_frame(self)

        self.title = Label(self.f_main, text='Matrix Rank Calculator', font=('Arial', 30, 'bold'),
                           bg=self.color_bg1, fg=self.color_text1)
        self.title.pack(pady=(30, 0))

        self.desc = Label(self.f_main, text='''the rank of a matrix A is the dimension 
of the vector space generated
(or spanned) by its columns. 
This corresponds to the maximal number 
of linearly independent columns of A.''', font=('Arial', 15), bg=self.color_bg1, fg=self.color_text1)
        self.desc.pack(pady=(30, 0))

        self.f_dims = LabelFrame(self.f_main, text='Custom Matrix', padx=100, pady=10, bg=self.color_bg1,
                                 fg=self.color_text3, relief=GROOVE)
        self.f_dims.pack(pady=(200, 50))

        self.dim_text = Label(self.f_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.dim_text.grid(row=0, column=0)

        self.dim_m = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim_m.current(0)
        self.dim_m.grid(row=0, column=1)

        self.X_text = Label(self.f_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.X_text.grid(row=0, column=2)

        self.dim_n = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim_n.current(0)
        self.dim_n.grid(row=0, column=3)

        self.set = Button(self.f_dims, text='Set matrix', bg=self.color_bg1, fg=self.color_text2,
                          activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                          command=lambda: GUI.set_matrix(self, 'rank', int(self.dim_m.get()), int(self.dim_n.get())))
        self.set.grid(row=0, column=6, padx=(100, 0))

        self.f_rand_dims = LabelFrame(self.f_main, text='Random Matrix', padx=100, pady=10, bg=self.color_bg1,
                                      fg=self.color_text3, relief=GROOVE)
        self.f_rand_dims.pack()

        self.rand_dim_text = Label(self.f_rand_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.rand_dim_text.grid(row=0, column=0)

        self.rand_dim_m = Entry(self.f_rand_dims, width=6)
        self.rand_dim_m.insert(0, '2')
        self.rand_dim_m.grid(row=0, column=1)

        self.rand_X_text = Label(self.f_rand_dims, text='X', bg=self.color_bg1, fg=self.color_text2)
        self.rand_X_text.grid(row=0, column=2)

        self.rand_dim_n = Entry(self.f_rand_dims, width=6)
        self.rand_dim_n.insert(0, '2')
        self.rand_dim_n.grid(row=0, column=3)

        self.rand_set = Button(self.f_rand_dims, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=3,
                               activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                               command=lambda: GUI.rand_calculate(self, 'rank', self.rand_dim_m.get(),
                                                                  self.rand_dim_n.get()))
        self.rand_set.grid(row=0, column=6, padx=(100, 0))

    def trace(self):
        GUI.clear_frame(self)

        self.title = Label(self.f_main, text='Matrix Trace Calculator', font=('Arial', 30, 'bold'),
                           bg=self.color_bg1, fg=self.color_text1)
        self.title.pack(pady=(30, 0))
        self.desc = Label(self.f_main, text='''In linear algebra, the trace of a square matrix A, denoted tr(A)
is defined to be the sum of elements on 
the main diagonal (from the upper left to the lower right) of A.
The trace of a matrix is the sum of its  eigenvalues
and it is invariant with respect to a change of basis.
This characterization can be used to define the trace of
a linear operator in general. 
The trace is only defined for a square matrix (n × n).''', font=('Arial', 15),  bg=self.color_bg1, fg=self.color_text1)
        self.desc.pack(pady=(30, 0))

        self.f_dims = LabelFrame(self.f_main, text='Custom Matrix', padx=100, pady=10, bg=self.color_bg1,
                                 fg=self.color_text3, relief=GROOVE)
        self.f_dims.pack(pady=(200, 50))

        self.dim_text = Label(self.f_dims, text='Matrix dimension:', bg=self.color_bg1, fg=self.color_text2)
        self.dim_text.grid(row=0, column=0)

        self.dim = ttk.Combobox(self.f_dims, values=self.dim_values, width=3, state='readonly')
        self.dim.current(0)
        self.dim.grid(row=0, column=1)

        self.set = Button(self.f_dims, text='Set matrix', bg=self.color_bg1, fg=self.color_text2,
                          activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                          command=lambda: GUI.set_matrix(self, 'trace', int(self.dim.get())))
        self.set.grid(row=0, column=6, padx=(100, 0))

        self.f_rand_dims = LabelFrame(self.f_main, text='Random Matrix', padx=100, pady=10, bg=self.color_bg1,
                                      fg=self.color_text3, relief=GROOVE)
        self.f_rand_dims.pack()

        self.rand_dim_text = Label(self.f_rand_dims, text='Matrix dimension:', bg=self.color_bg1,
                                   fg=self.color_text2)
        self.rand_dim_text.grid(row=0, column=0)

        self.rand_dim = Entry(self.f_rand_dims, width=6)
        self.rand_dim.insert(0, '2')
        self.rand_dim.grid(row=0, column=1)

        self.rand_set = Button(self.f_rand_dims, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=3,
                               activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                               command=lambda: GUI.rand_calculate(self, 'trace', self.rand_dim.get()))
        self.rand_set.grid(row=0, column=6, padx=(100, 0))

    def clear_frame(self):
        """This method clears all the widgets from the main (left) frame"""
        for widget in self.f_main.winfo_children():
            widget.destroy()

    def set_matrix(self, func, a, b=0, c=0):
        """This method opens a new window for the matrix (or matrices) input,
        according to the desired calculation"""
        self.input_win = Toplevel(root, bg=self.color_bg1)
        self.input_win.iconbitmap('matrix_ico.ico')
        self.input_win.title('Matrix Calculator')

        if func == 'add_sub':
            # According to the number of matrices needed to be inputted and the dimensions of these
            # matrices, the correct numbers of frames are created (f_up, f_down, f1, f1_grid etc.)
            # as well as entry boxes and some auxiliary buttons ("Clear", "Fill with 1's" etc.)
            #
            # Similar process is done on all other 'elif' conditions,
            # so commenting will only be present on this 'if' block

            self.input_win.resizable(False, False)

            # Creating the frames and other widgets
            self.f_up = Frame(self.input_win, bg=self.color_bg1)
            self.f_up.pack(pady=(20, 0))
            self.f1 = Frame(self.f_up, bg=self.color_bg1)
            self.f1.pack(side='left', padx=25)
            self.separator = ttk.Separator(self.f_up, orient='vertical')
            self.separator.pack(side='left', padx=20, pady=20, fill='y')
            self.f2 = Frame(self.f_up, bg=self.color_bg1)
            self.f2.pack(side='left', padx=25)
            self.f_down = Frame(self.input_win, bg=self.color_bg1)
            self.f_down.pack(pady=(15, 20))
            self.calculate = Button(self.f_down, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=30,
                                    pady=5, font=('Arial', 15), activebackground=self.color_bg1,
                                    activeforeground=self.color_text2, relief=RIDGE,
                                    command=lambda: GUI.calculate(self, func, a, b))
            self.calculate.pack()

            self.matrix_A_text = Label(self.f1, text='Input matrix A:', bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 10))
            self.matrix_A_text.pack()
            self.matrix_B_text = Label(self.f2, text='Input matrix B:', bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 10))
            self.matrix_B_text.pack()

            self.f1_grid = Frame(self.f1, bg=self.color_bg1)
            self.f1_grid.pack()
            self.f1_buttons = Frame(self.f1, bg=self.color_bg1)
            self.f1_buttons.pack(pady=15)

            self.f2_grid = Frame(self.f2, bg=self.color_bg1)
            self.f2_grid.pack()
            self.f2_buttons = Frame(self.f2, bg=self.color_bg1)
            self.f2_buttons.pack(pady=15)

            # Creating empty 2D lists, which will later contain the entry widgets
            self.matrix_A_entries = [[] for _ in range(a)]
            self.matrix_B_entries = [[] for _ in range(a)]

            for j in range(b):  # This 'for' loop displays numbers above entry boxes
                Label(self.f1_grid, text=j+1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
                Label(self.f2_grid, text=j + 1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
            for i in range(a):  # This 'for' loop displays the entry boxes and numbers left to them
                Label(self.f1_grid, text=i+1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                Label(self.f2_grid, text=i+1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                for j in range(b):
                    x = Entry(self.f1_grid, width=5)
                    x.grid(row=i+1, column=j+1, padx=3, pady=3)
                    self.matrix_A_entries[i].append(x)

                    y = Entry(self.f2_grid, width=5)
                    y.grid(row=i+1, column=j+1, padx=3, pady=3)
                    self.matrix_B_entries[i].append(y)

            # Creating and displaying the auxiliary buttons
            self.clear_button_A = Button(self.f1_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                         font=('Arial', 8), activebackground=self.color_bg1,
                                         activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                         command=lambda: GUI.clear_cells(self, self.matrix_A_entries))
            self.clear_button_A.grid(row=0, column=0, columnspan=3, pady=3)

            self.zeros_button_A = Button(self.f1_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                         font=('Arial', 8), activebackground=self.color_bg1,
                                         activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                         command=lambda: GUI.fill_zeros(self, self.matrix_A_entries))
            self.zeros_button_A.grid(row=1, column=0, pady=3)

            self.ones_button_A = Button(self.f1_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                        font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                        command=lambda: GUI.fill_ones(self, self.matrix_A_entries))
            self.ones_button_A.grid(row=1, column=1, pady=3)

            self.mem_sv_button_A = Button(self.f1_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                          fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                          activeforeground=self.color_text2, relief=RIDGE,
                                          command=lambda: GUI.mem_sv(self, self.matrix_A_entries))
            self.mem_sv_button_A.grid(row=2, column=0, padx=3, pady=3)

            self.mem_ld_button_A = Button(self.f1_buttons, text="Load from memory", bg=self.color_bg1,
                                          fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                          activeforeground=self.color_text2, relief=RIDGE,
                                          command=lambda: GUI.mem_ld(self, self.matrix_A_entries))
            self.mem_ld_button_A.grid(row=2, column=1, padx=3, pady=3)

            self.clear_button_B = Button(self.f2_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                         font=('Arial', 8), activebackground=self.color_bg1,
                                         activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                         command=lambda: GUI.clear_cells(self, self.matrix_B_entries))
            self.clear_button_B.grid(row=0, column=0, columnspan=3, pady=3)

            self.zeros_button_B = Button(self.f2_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                         font=('Arial', 8), activebackground=self.color_bg1,
                                         activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                         command=lambda: GUI.fill_zeros(self, self.matrix_B_entries))
            self.zeros_button_B.grid(row=1, column=0, pady=3)

            self.ones_button_B = Button(self.f2_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                        font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                        command=lambda: GUI.fill_ones(self, self.matrix_B_entries))
            self.ones_button_B.grid(row=1, column=1, pady=3)

            self.mem_sv_button_B = Button(self.f2_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                          fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                          activeforeground=self.color_text2, relief=RIDGE,
                                          command=lambda: GUI.mem_sv(self, self.matrix_B_entries))
            self.mem_sv_button_B.grid(row=2, column=0, padx=3, pady=3)

            self.mem_ld_button_B = Button(self.f2_buttons, text="Load from memory", bg=self.color_bg1,
                                          fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                          activeforeground=self.color_text2, relief=RIDGE,
                                          command=lambda: GUI.mem_ld(self, self.matrix_B_entries))
            self.mem_ld_button_B.grid(row=2, column=1, padx=3, pady=3)

        elif func == 'mul_num':
            is_error = False

            if self.num_entry.get():
                try:
                    float(self.num_entry.get())
                except ValueError:
                    is_error = True
                    self.input_win.destroy()
                    self.errors('num')

            if not is_error:
                self.input_win.resizable(False, False)

                self.f = Frame(self.input_win, bg=self.color_bg1)
                self.f.pack(pady=20)

                self.matrix_text = Label(self.f, text='Input matrix:', bg=self.color_bg1, fg=self.color_text2,
                                         font=('Arial', 10))
                self.matrix_text.pack()

                self.f_grid = Frame(self.f, bg=self.color_bg1)
                self.f_grid.pack(padx=25)
                self.f_buttons = Frame(self.f, bg=self.color_bg1)
                self.f_buttons.pack(pady=(15, 0), padx=25)

                self.matrix_entries = [[] for _ in range(a)]
                for j in range(b):
                    Label(self.f_grid, text=j+1, font=('Arial', 8), bg=self.color_bg1,
                          fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
                for i in range(a):
                    Label(self.f_grid, text=i + 1, font=('Arial', 8), bg=self.color_bg1,
                          fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                    for j in range(b):
                        x = Entry(self.f_grid, width=5)
                        x.grid(row=i+1, column=j+1, padx=3, pady=3)
                        self.matrix_entries[i].append(x)

                self.clear_button = Button(self.f_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                           font=('Arial', 8), activebackground=self.color_bg1,
                                           activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                           command=lambda: GUI.clear_cells(self, self.matrix_entries))
                self.clear_button.grid(row=0, column=0, columnspan=3, pady=3)

                self.zeros_button = Button(self.f_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                           font=('Arial', 8), activebackground=self.color_bg1,
                                           activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                           command=lambda: GUI.fill_zeros(self, self.matrix_entries))
                self.zeros_button.grid(row=1, column=0, pady=3)

                self.ones_button = Button(self.f_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                          font=('Arial', 8),
                                          activebackground=self.color_bg1, activeforeground=self.color_text2, padx=20,
                                          relief=RIDGE, command=lambda: GUI.fill_ones(self, self.matrix_entries))
                self.ones_button.grid(row=1, column=1, pady=3)

                self.mem_sv_button = Button(self.f_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                            fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                            activeforeground=self.color_text2, relief=RIDGE,
                                            command=lambda: GUI.mem_sv(self, self.matrix_entries))
                self.mem_sv_button.grid(row=2, column=0, padx=3, pady=3)

                self.mem_ld_button = Button(self.f_buttons, text="Load from memory", bg=self.color_bg1,
                                            fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                            activeforeground=self.color_text2, relief=RIDGE,
                                            command=lambda: GUI.mem_ld(self, self.matrix_entries))
                self.mem_ld_button.grid(row=2, column=1, padx=3, pady=3)

                self.calculate = Button(self.f_buttons, text='Calculate', bg=self.color_bg1, fg=self.color_text2,
                                        padx=30, pady=5, font=('Arial', 15), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.calculate(self, func, a, b))
                self.calculate.grid(row=3, column=0, columnspan=3, pady=(30, 0))

        elif func == 'mul':
            self.input_win.resizable(False, False)

            self.f_up = Frame(self.input_win, bg=self.color_bg1)
            self.f_up.pack(pady=(20, 0))
            self.f1 = Frame(self.f_up, bg=self.color_bg1)
            self.f1.pack(side='left', padx=25)
            self.separator = ttk.Separator(self.f_up, orient='vertical')
            self.separator.pack(side='left', padx=20, pady=20, fill='y')
            self.f2 = Frame(self.f_up, bg=self.color_bg1)
            self.f2.pack(side='left', padx=25)
            self.f_down = Frame(self.input_win, bg=self.color_bg1)
            self.f_down.pack(pady=(15, 20))
            self.calculate = Button(self.f_down, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=30,
                                    pady=5, font=('Arial', 15), activebackground=self.color_bg1,
                                    activeforeground=self.color_text2, relief=RIDGE,
                                    command=lambda: GUI.calculate(self, func, a, b, c))
            self.calculate.pack()

            self.matrix_A_text = Label(self.f1, text='Input matrix A:', bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 10))
            self.matrix_A_text.pack()
            self.matrix_B_text = Label(self.f2, text='Input matrix B:', bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 10))
            self.matrix_B_text.pack()

            self.f1_grid = Frame(self.f1, bg=self.color_bg1)
            self.f1_grid.pack()
            self.f1_buttons = Frame(self.f1, bg=self.color_bg1)
            self.f1_buttons.pack(pady=15)

            self.f2_grid = Frame(self.f2, bg=self.color_bg1)
            self.f2_grid.pack()
            self.f2_buttons = Frame(self.f2, bg=self.color_bg1)
            self.f2_buttons.pack(pady=15)

            self.matrix_A_entries = [[] for _ in range(a)]
            self.matrix_B_entries = [[] for _ in range(b)]

            for j in range(b):
                Label(self.f1_grid, text=j+1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
            for i in range(a):
                Label(self.f1_grid, text=i + 1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                for j in range(b):
                    x = Entry(self.f1_grid, width=5)
                    x.grid(row=i+1, column=j+1, padx=3, pady=3)
                    self.matrix_A_entries[i].append(x)

            for j in range(c):
                Label(self.f2_grid, text=j+1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
            for i in range(b):
                Label(self.f2_grid, text=i + 1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                for j in range(c):
                    y = Entry(self.f2_grid, width=5)
                    y.grid(row=i+1, column=j+1, padx=3, pady=3)
                    self.matrix_B_entries[i].append(y)

            self.clear_button_A = Button(self.f1_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                         font=('Arial', 8), activebackground=self.color_bg1,
                                         activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                         command=lambda: GUI.clear_cells(self, self.matrix_A_entries))
            self.clear_button_A.grid(row=0, column=0, columnspan=3, pady=3)

            self.zeros_button_A = Button(self.f1_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                         font=('Arial', 8), activebackground=self.color_bg1,
                                         activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                         command=lambda: GUI.fill_zeros(self, self.matrix_A_entries))
            self.zeros_button_A.grid(row=1, column=0, pady=3)

            self.ones_button_A = Button(self.f1_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                        font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                        command=lambda: GUI.fill_ones(self, self.matrix_A_entries))
            self.ones_button_A.grid(row=1, column=1, pady=3)

            self.mem_sv_button_A = Button(self.f1_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                          fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                          activeforeground=self.color_text2, relief=RIDGE,
                                          command=lambda: GUI.mem_sv(self, self.matrix_A_entries))
            self.mem_sv_button_A.grid(row=2, column=0, padx=3, pady=3)

            self.mem_ld_button_A = Button(self.f1_buttons, text="Load from memory", bg=self.color_bg1,
                                          fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                          activeforeground=self.color_text2, relief=RIDGE,
                                          command=lambda: GUI.mem_ld(self, self.matrix_A_entries))
            self.mem_ld_button_A.grid(row=2, column=1, padx=3, pady=3)

            self.clear_button_B = Button(self.f2_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                         font=('Arial', 8), activebackground=self.color_bg1,
                                         activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                         command=lambda: GUI.clear_cells(self, self.matrix_B_entries))
            self.clear_button_B.grid(row=0, column=0, columnspan=3, pady=3)

            self.zeros_button_B = Button(self.f2_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                         font=('Arial', 8), activebackground=self.color_bg1,
                                         activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                         command=lambda: GUI.fill_zeros(self, self.matrix_B_entries))
            self.zeros_button_B.grid(row=1, column=0, pady=3)

            self.ones_button_B = Button(self.f2_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                        font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                        command=lambda: GUI.fill_ones(self, self.matrix_B_entries))
            self.ones_button_B.grid(row=1, column=1, pady=3)

            self.mem_sv_button_B = Button(self.f2_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                          fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                          activeforeground=self.color_text2, relief=RIDGE,
                                          command=lambda: GUI.mem_sv(self, self.matrix_B_entries))
            self.mem_sv_button_B.grid(row=2, column=0, padx=3, pady=3)

            self.mem_ld_button_B = Button(self.f2_buttons, text="Load from memory", bg=self.color_bg1,
                                          fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                          activeforeground=self.color_text2, relief=RIDGE,
                                          command=lambda: GUI.mem_ld(self, self.matrix_B_entries))
            self.mem_ld_button_B.grid(row=2, column=1, padx=3, pady=3)

        elif func == 'power':
            is_error = False
            if self.power_entry.get():
                if not self.power_entry.get().isdigit():
                    self.input_win.destroy()
                    is_error = True
                    self.errors('power')

            if not is_error:
                self.input_win.resizable(False, False)

                self.f = Frame(self.input_win, bg=self.color_bg1)
                self.f.pack(pady=20)

                self.matrix_text = Label(self.f, text='Input matrix:', bg=self.color_bg1, fg=self.color_text2,
                                         font=('Arial', 10))
                self.matrix_text.pack()

                self.f_grid = Frame(self.f, bg=self.color_bg1)
                self.f_grid.pack(padx=25)
                self.f_buttons = Frame(self.f, bg=self.color_bg1)
                self.f_buttons.pack(pady=(15, 0), padx=25)

                self.matrix_entries = [[] for _ in range(a)]

                for j in range(a):
                    Label(self.f_grid, text=j+1, font=('Arial', 8), bg=self.color_bg1,
                          fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
                for i in range(a):
                    Label(self.f_grid, text=i + 1, font=('Arial', 8), bg=self.color_bg1,
                          fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                    for j in range(a):
                        x = Entry(self.f_grid, width=5)
                        x.grid(row=i+1, column=j+1, padx=3, pady=3)
                        self.matrix_entries[i].append(x)

                self.clear_button = Button(self.f_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                           font=('Arial', 8), activebackground=self.color_bg1,
                                           activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                           command=lambda: GUI.clear_cells(self, self.matrix_entries))
                self.clear_button.grid(row=0, column=0, columnspan=3, pady=3)

                self.zeros_button = Button(self.f_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                           font=('Arial', 8), activebackground=self.color_bg1,
                                           activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                           command=lambda: GUI.fill_zeros(self, self.matrix_entries))
                self.zeros_button.grid(row=1, column=0, pady=3)

                self.ones_button = Button(self.f_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                          font=('Arial', 8),
                                          activebackground=self.color_bg1, activeforeground=self.color_text2, padx=20,
                                          relief=RIDGE, command=lambda: GUI.fill_ones(self, self.matrix_entries))
                self.ones_button.grid(row=1, column=1, pady=3)

                self.mem_sv_button = Button(self.f_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                            fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                            activeforeground=self.color_text2, relief=RIDGE,
                                            command=lambda: GUI.mem_sv(self, self.matrix_entries))
                self.mem_sv_button.grid(row=2, column=0, padx=3, pady=3)

                self.mem_ld_button = Button(self.f_buttons, text="Load from memory", bg=self.color_bg1,
                                            fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                            activeforeground=self.color_text2, relief=RIDGE,
                                            command=lambda: GUI.mem_ld(self, self.matrix_entries))
                self.mem_ld_button.grid(row=2, column=1, padx=3, pady=3)

                self.calculate = Button(self.f_buttons, text='Calculate', bg=self.color_bg1, fg=self.color_text2,
                                        padx=30, pady=5, font=('Arial', 15), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.calculate(self, func, a, b))
                self.calculate.grid(row=3, column=0, columnspan=3, pady=(30, 0))

        elif func == 'det':
            self.input_win.resizable(False, False)

            self.f = Frame(self.input_win, bg=self.color_bg1)
            self.f.pack(pady=20)

            self.matrix_text = Label(self.f, text='Input matrix:', bg=self.color_bg1, fg=self.color_text2,
                                     font=('Arial', 10))
            self.matrix_text.pack()

            self.f_grid = Frame(self.f, bg=self.color_bg1)
            self.f_grid.pack(padx=25)
            self.f_buttons = Frame(self.f, bg=self.color_bg1)
            self.f_buttons.pack(pady=(15, 0), padx=25)

            self.matrix_entries = [[] for _ in range(a)]

            for j in range(a):
                Label(self.f_grid, text=j+1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
            for i in range(a):
                Label(self.f_grid, text=i + 1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                for j in range(a):
                    x = Entry(self.f_grid, width=5)
                    x.grid(row=i+1, column=j+1, padx=3, pady=3)
                    self.matrix_entries[i].append(x)

            self.clear_button = Button(self.f_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 8), activebackground=self.color_bg1,
                                       activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                       command=lambda: GUI.clear_cells(self, self.matrix_entries))
            self.clear_button.grid(row=0, column=0, columnspan=3, pady=3)

            self.zeros_button = Button(self.f_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 8), activebackground=self.color_bg1,
                                       activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                       command=lambda: GUI.fill_zeros(self, self.matrix_entries))
            self.zeros_button.grid(row=1, column=0, pady=3)

            self.ones_button = Button(self.f_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                      font=('Arial', 8),
                                      activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                                      padx=20, command=lambda: GUI.fill_ones(self, self.matrix_entries))
            self.ones_button.grid(row=1, column=1, pady=3)

            self.mem_sv_button = Button(self.f_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                        fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.mem_sv(self, self.matrix_entries))
            self.mem_sv_button.grid(row=2, column=0, padx=3, pady=3)

            self.mem_ld_button = Button(self.f_buttons, text="Load from memory", bg=self.color_bg1,
                                        fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.mem_ld(self, self.matrix_entries))
            self.mem_ld_button.grid(row=2, column=1, padx=3, pady=3)

            self.calculate = Button(self.f_buttons, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=30,
                                    pady=5, font=('Arial', 15), activebackground=self.color_bg1,
                                    activeforeground=self.color_text2, relief=RIDGE,
                                    command=lambda: GUI.calculate(self, func, a, b))
            self.calculate.grid(row=3, column=0, columnspan=3, pady=(30, 0))

        elif func == 'inv':
            self.input_win.resizable(False, False)

            self.f = Frame(self.input_win, bg=self.color_bg1)
            self.f.pack(pady=20)

            self.matrix_text = Label(self.f, text='Input matrix:', bg=self.color_bg1, fg=self.color_text2,
                                     font=('Arial', 10))
            self.matrix_text.pack()

            self.f_grid = Frame(self.f, bg=self.color_bg1)
            self.f_grid.pack(padx=25)
            self.f_buttons = Frame(self.f, bg=self.color_bg1)
            self.f_buttons.pack(pady=(15, 0), padx=25)

            self.matrix_entries = [[] for _ in range(a)]

            for j in range(a):
                Label(self.f_grid, text=j+1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
            for i in range(a):
                Label(self.f_grid, text=i + 1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                for j in range(a):
                    x = Entry(self.f_grid, width=5)
                    x.grid(row=i+1, column=j+1, padx=3, pady=3)
                    self.matrix_entries[i].append(x)

            self.clear_button = Button(self.f_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 8), activebackground=self.color_bg1,
                                       activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                       command=lambda: GUI.clear_cells(self, self.matrix_entries))
            self.clear_button.grid(row=0, column=0, columnspan=3, pady=3)

            self.zeros_button = Button(self.f_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 8), activebackground=self.color_bg1,
                                       activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                       command=lambda: GUI.fill_zeros(self, self.matrix_entries))
            self.zeros_button.grid(row=1, column=0, pady=3)

            self.ones_button = Button(self.f_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                      font=('Arial', 8),
                                      activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                                      padx=20, command=lambda: GUI.fill_ones(self, self.matrix_entries))
            self.ones_button.grid(row=1, column=1, pady=3)

            self.mem_sv_button = Button(self.f_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                        fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.mem_sv(self, self.matrix_entries))
            self.mem_sv_button.grid(row=2, column=0, padx=3, pady=3)

            self.mem_ld_button = Button(self.f_buttons, text="Load from memory", bg=self.color_bg1,
                                        fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.mem_ld(self, self.matrix_entries))
            self.mem_ld_button.grid(row=2, column=1, padx=3, pady=3)

            self.calculate = Button(self.f_buttons, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=30,
                                    pady=5, font=('Arial', 15), activebackground=self.color_bg1,
                                    activeforeground=self.color_text2, relief=RIDGE,
                                    command=lambda: GUI.calculate(self, func, a, b))
            self.calculate.grid(row=3, column=0, columnspan=3, pady=(30, 0))

        elif func == 'trans':
            self.input_win.resizable(False, False)

            self.f = Frame(self.input_win, bg=self.color_bg1)
            self.f.pack(pady=20)

            self.matrix_text = Label(self.f, text='Input matrix:', bg=self.color_bg1, fg=self.color_text2,
                                     font=('Arial', 10))
            self.matrix_text.pack()

            self.f_grid = Frame(self.f, bg=self.color_bg1)
            self.f_grid.pack(padx=25)
            self.f_buttons = Frame(self.f, bg=self.color_bg1)
            self.f_buttons.pack(pady=(15, 0), padx=25)

            self.matrix_entries = [[] for _ in range(a)]

            for j in range(b):
                Label(self.f_grid, text=j+1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
            for i in range(a):
                Label(self.f_grid, text=i + 1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                for j in range(b):
                    x = Entry(self.f_grid, width=5)
                    x.grid(row=i+1, column=j+1, padx=3, pady=3)
                    self.matrix_entries[i].append(x)

            self.clear_button = Button(self.f_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 8), activebackground=self.color_bg1,
                                       activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                       command=lambda: GUI.clear_cells(self, self.matrix_entries))
            self.clear_button.grid(row=0, column=0, columnspan=3, pady=3)

            self.zeros_button = Button(self.f_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 8), activebackground=self.color_bg1,
                                       activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                       command=lambda: GUI.fill_zeros(self, self.matrix_entries))
            self.zeros_button.grid(row=1, column=0, pady=3)

            self.ones_button = Button(self.f_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                      font=('Arial', 8),
                                      activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                                      padx=20, command=lambda: GUI.fill_ones(self, self.matrix_entries))
            self.ones_button.grid(row=1, column=1, pady=3)

            self.mem_sv_button = Button(self.f_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                        fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.mem_sv(self, self.matrix_entries))
            self.mem_sv_button.grid(row=2, column=0, padx=3, pady=3)

            self.mem_ld_button = Button(self.f_buttons, text="Load from memory", bg=self.color_bg1,
                                        fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.mem_ld(self, self.matrix_entries))
            self.mem_ld_button.grid(row=2, column=1, padx=3, pady=3)

            self.calculate = Button(self.f_buttons, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=30,
                                    pady=5, font=('Arial', 15), activebackground=self.color_bg1,
                                    activeforeground=self.color_text2, relief=RIDGE,
                                    command=lambda: GUI.calculate(self, func, a, b))
            self.calculate.grid(row=3, column=0, columnspan=3, pady=(30, 0))

        elif func == 'rank':
            self.input_win.resizable(False, False)

            self.f = Frame(self.input_win, bg=self.color_bg1)
            self.f.pack(pady=20)

            self.matrix_text = Label(self.f, text='Input matrix:', bg=self.color_bg1, fg=self.color_text2,
                                     font=('Arial', 10))
            self.matrix_text.pack()

            self.f_grid = Frame(self.f, bg=self.color_bg1)
            self.f_grid.pack(padx=25)
            self.f_buttons = Frame(self.f, bg=self.color_bg1)
            self.f_buttons.pack(pady=(15, 0), padx=25)

            self.matrix_entries = [[] for _ in range(a)]

            for j in range(b):
                Label(self.f_grid, text=j+1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
            for i in range(a):
                Label(self.f_grid, text=i + 1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                for j in range(b):
                    x = Entry(self.f_grid, width=5)
                    x.grid(row=i+1, column=j+1, padx=3, pady=3)
                    self.matrix_entries[i].append(x)

            self.clear_button = Button(self.f_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 8), activebackground=self.color_bg1,
                                       activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                       command=lambda: GUI.clear_cells(self, self.matrix_entries))
            self.clear_button.grid(row=0, column=0, columnspan=3, pady=3)

            self.zeros_button = Button(self.f_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 8), activebackground=self.color_bg1,
                                       activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                       command=lambda: GUI.fill_zeros(self, self.matrix_entries))
            self.zeros_button.grid(row=1, column=0, pady=3)

            self.ones_button = Button(self.f_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                      font=('Arial', 8),
                                      activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                                      padx=20, command=lambda: GUI.fill_ones(self, self.matrix_entries))
            self.ones_button.grid(row=1, column=1, pady=3)

            self.mem_sv_button = Button(self.f_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                        fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.mem_sv(self, self.matrix_entries))
            self.mem_sv_button.grid(row=2, column=0, padx=3, pady=3)

            self.mem_ld_button = Button(self.f_buttons, text="Load from memory", bg=self.color_bg1,
                                        fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.mem_ld(self, self.matrix_entries))
            self.mem_ld_button.grid(row=2, column=1, padx=3, pady=3)

            self.calculate = Button(self.f_buttons, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=30,
                                    pady=5, font=('Arial', 15), activebackground=self.color_bg1,
                                    activeforeground=self.color_text2, relief=RIDGE,
                                    command=lambda: GUI.calculate(self, func, a, b))
            self.calculate.grid(row=3, column=0, columnspan=3, pady=(30, 0))

        elif func == 'trace':
            self.input_win.resizable(False, False)

            self.f = Frame(self.input_win, bg=self.color_bg1)
            self.f.pack(pady=20)

            self.matrix_text = Label(self.f, text='Input matrix:', bg=self.color_bg1, fg=self.color_text2,
                                     font=('Arial', 10))
            self.matrix_text.pack()

            self.f_grid = Frame(self.f, bg=self.color_bg1)
            self.f_grid.pack(padx=25)
            self.f_buttons = Frame(self.f, bg=self.color_bg1)
            self.f_buttons.pack(pady=(15, 0), padx=25)

            self.matrix_entries = [[] for _ in range(a)]

            for j in range(a):
                Label(self.f_grid, text=j + 1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=0, column=j + 1, padx=(3, 0))
            for i in range(a):
                Label(self.f_grid, text=i + 1, font=('Arial', 8), bg=self.color_bg1,
                      fg=self.color_text2).grid(row=i + 1, column=0, padx=(3, 0))
                for j in range(a):
                    x = Entry(self.f_grid, width=5)
                    x.grid(row=i + 1, column=j + 1, padx=3, pady=3)
                    self.matrix_entries[i].append(x)

            self.clear_button = Button(self.f_buttons, text='Clear', bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 8), activebackground=self.color_bg1,
                                       activeforeground=self.color_text2, relief=RIDGE, padx=33,
                                       command=lambda: GUI.clear_cells(self, self.matrix_entries))
            self.clear_button.grid(row=0, column=0, columnspan=3, pady=3)

            self.zeros_button = Button(self.f_buttons, text="Fill with 0's", bg=self.color_bg1, fg=self.color_text2,
                                       font=('Arial', 8), activebackground=self.color_bg1,
                                       activeforeground=self.color_text2, relief=RIDGE, padx=20,
                                       command=lambda: GUI.fill_zeros(self, self.matrix_entries))
            self.zeros_button.grid(row=1, column=0, pady=3)

            self.ones_button = Button(self.f_buttons, text="Fill with 1's", bg=self.color_bg1, fg=self.color_text2,
                                      font=('Arial', 8),
                                      activebackground=self.color_bg1, activeforeground=self.color_text2, relief=RIDGE,
                                      padx=20, command=lambda: GUI.fill_ones(self, self.matrix_entries))
            self.ones_button.grid(row=1, column=1, pady=3)

            self.mem_sv_button = Button(self.f_buttons, text="  Save to memory  ", bg=self.color_bg1,
                                        fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.mem_sv(self, self.matrix_entries))
            self.mem_sv_button.grid(row=2, column=0, padx=3, pady=3)

            self.mem_ld_button = Button(self.f_buttons, text="Load from memory", bg=self.color_bg1,
                                        fg=self.color_text2, font=('Arial', 8), activebackground=self.color_bg1,
                                        activeforeground=self.color_text2, relief=RIDGE,
                                        command=lambda: GUI.mem_ld(self, self.matrix_entries))
            self.mem_ld_button.grid(row=2, column=1, padx=3, pady=3)

            self.calculate = Button(self.f_buttons, text='Calculate', bg=self.color_bg1, fg=self.color_text2, padx=30,
                                    pady=5, font=('Arial', 15), activebackground=self.color_bg1,
                                    activeforeground=self.color_text2, relief=RIDGE,
                                    command=lambda: GUI.calculate(self, func, a, b))
            self.calculate.grid(row=3, column=0, columnspan=3, pady=(30, 0))

    def calculate(self, func, a, b=0, c=0):
        """This method passes the matrix values in NumPy arrays and
        proceeds with the according calculation"""
        is_error = False  # 'True' if an error has occurred

        if func == 'add_sub':
            self.matrix_A = np.zeros((a, b))
            self.matrix_B = np.zeros((a, b))

            try:
                for i in range(a):
                    for j in range(b):
                        if self.matrix_A_entries[i][j].get() == '':
                            self.matrix_A_entries[i][j].insert(0, '0')
                        if self.matrix_B_entries[i][j].get() == '':
                            self.matrix_B_entries[i][j].insert(0, '0')

                        self.matrix_A[i, j] = float(self.matrix_A_entries[i][j].get())
                        self.matrix_B[i, j] = float(self.matrix_B_entries[i][j].get())
            except ValueError:
                is_error = True
                self.errors('alpha')
            except:
                is_error = True
                self.errors('unexpected')

            if not is_error:
                print("Matrix A:", self.matrix_A, sep="\n")
                print()
                print("Matrix B:", self.matrix_B, sep="\n")
                print()

                if self.op.get() == '+':
                    start = time.perf_counter()
                    calc = SimpleCalculation.matrix_add(self.matrix_A, self.matrix_B)
                    finish = time.perf_counter()

                elif self.op.get() == '-':
                    start = time.perf_counter()
                    calc = SimpleCalculation.matrix_sub(self.matrix_A, self.matrix_B)
                    finish = time.perf_counter()

                self.time = round(finish - start, 3)
                print("Result: ", calc, sep="\n")
                print()
                print("Time:", self.time)
                print()
                self.result_show(calc)

        elif func == 'mul_num':
            self.matrix = np.zeros((a, b))

            try:
                for i in range(a):
                    for j in range(b):
                        if self.matrix_entries[i][j].get() == '':
                            self.matrix_entries[i][j].insert(0, '0')

                        self.matrix[i, j] = float(self.matrix_entries[i][j].get())
            except ValueError:
                is_error = True
                self.errors('alpha')
            except:
                is_error = True
                self.errors('unexpected')

            if not is_error:
                print("Matrix:", self.matrix, sep="\n")
                print()

                if not self.num_entry.get(): num = 1
                else: num = self.num_entry.get()

                start = time.perf_counter()
                calc = SimpleCalculation.matrix_mul_num(self.matrix, num)
                finish = time.perf_counter()

                self.time = round(finish - start, 3)
                print("Result:", calc, sep="\n")
                print()
                print("Time:", self.time)
                print()
                self.result_show(calc)

        elif func == 'mul':
            self.matrix_A = np.zeros((a, b))
            self.matrix_B = np.zeros((b, c))

            try:
                for i in range(a):
                    for j in range(b):
                        if self.matrix_A_entries[i][j].get() == '':
                            self.matrix_A_entries[i][j].insert(0, '0')

                        self.matrix_A[i, j] = float(self.matrix_A_entries[i][j].get())
                for i in range(b):
                    for j in range(c):
                        if self.matrix_B_entries[i][j].get() == '':
                            self.matrix_B_entries[i][j].insert(0, '0')

                        self.matrix_B[i, j] = float(self.matrix_B_entries[i][j].get())
            except ValueError:
                is_error = True
                self.errors('alpha')
            except:
                is_error = True
                self.errors('unexpected')

            if not is_error:
                print("Matrix A:", self.matrix_A, sep="\n")
                print()
                print("Matrix B:", self.matrix_B, sep="\n")
                print()

                start = time.perf_counter()
                calc = SimpleCalculation.matrix_mul(self.matrix_A, self.matrix_B)
                finish = time.perf_counter()

                self.time = round(finish - start, 3)
                print("Result:", calc, sep="\n")
                print()
                print("Time:", self.time)
                print()
                self.result_show(calc)

        elif func == 'power':
            self.matrix = np.zeros((a, a))

            try:
                for i in range(a):
                    for j in range(a):
                        if self.matrix_entries[i][j].get() == '':
                            self.matrix_entries[i][j].insert(0, '0')

                        self.matrix[i, j] = float(self.matrix_entries[i][j].get())
            except ValueError:
                is_error = True
                self.errors('alpha')
            except:
                is_error = True
                self.errors('unexpected')

            if not is_error:
                print("Matrix:", self.matrix, sep="\n")
                print()

                if not self.power_entry.get(): num = 1
                else: num = int(self.power_entry.get())

                start = time.perf_counter()
                calc = SimpleCalculation.matrix_power(self.matrix, num)
                finish = time.perf_counter()

                self.time = round(finish - start, 3)
                print("Result:", calc, sep="\n")
                print()
                print("Time:", self.time)
                print()
                self.result_show(calc)

        elif func == 'det':
            self.matrix = np.zeros((a, a))

            try:
                for i in range(a):
                    for j in range(a):
                        if self.matrix_entries[i][j].get() == '':
                            self.matrix_entries[i][j].insert(0, '0')

                        self.matrix[i, j] = float(self.matrix_entries[i][j].get())
            except ValueError:
                is_error = True
                self.errors('alpha')
            except:
                is_error = True
                self.errors('unexpected')

            if not is_error:
                print("Matrix:",self.matrix, sep="\n")
                print()

                start = time.perf_counter()
                calc = SimpleCalculation.matrix_det(self.matrix)
                finish = time.perf_counter()

                self.time = round(finish - start, 3)
                print("Result:", calc, sep="\n")
                print()
                print("time:", self.time)
                print()
                self.result_show(calc, 'det')

        elif func == 'inv':
            self.matrix = np.zeros((a, a))

            try:
                for i in range(a):
                    for j in range(a):
                        if self.matrix_entries[i][j].get() == '':
                            self.matrix_entries[i][j].insert(0, '0')

                        self.matrix[i, j] = float(self.matrix_entries[i][j].get())
            except ValueError:
                is_error = True
                self.errors('alpha')
            except:
                is_error = True
                self.errors('unexpected')

            if not is_error:
                print("Matrix:", self.matrix, sep="\n")
                print()

                try:
                    start = time.perf_counter()
                    calc = SimpleCalculation.matrix_inv(self.matrix)
                    finish = time.perf_counter()
                except np.linalg.LinAlgError:
                    is_error = True
                    self.errors('singular')

                if not is_error:
                    self.time = round(finish - start, 3)
                    print("Result:", calc, sep="\n")
                    print()
                    print("Time:", self.time)
                    print()
                    self.result_show(calc)

        elif func == 'trans':
            self.matrix = np.zeros((a, b))
            try:
                for i in range(a):
                    for j in range(b):
                        if self.matrix_entries[i][j].get() == '':
                            self.matrix_entries[i][j].insert(0, '0')

                        self.matrix[i, j] = float(self.matrix_entries[i][j].get())
            except ValueError:
                is_error = True
                self.errors('alpha')
            except:
                is_error = True
                self.errors('unexpected')

            if not is_error:
                print("Matrix:", self.matrix, sep="\n")
                print()

                start = time.perf_counter()
                calc = SimpleCalculation.matrix_trans(self.matrix)
                finish = time.perf_counter()

                self.time = round(finish - start, 3)
                print("Result:", calc)
                print()
                print("Time:", self.time)
                print()
                self.result_show(calc)

        elif func == 'rank':
            self.matrix = np.zeros((a, b))
            try:
                for i in range(a):
                    for j in range(b):
                        if self.matrix_entries[i][j].get() == '':
                            self.matrix_entries[i][j].insert(0, '0')

                        self.matrix[i, j] = float(self.matrix_entries[i][j].get())
            except ValueError:
                is_error = True
                self.errors('alpha')
            except:
                is_error = True
                self.errors('unexpected')

            if not is_error:
                print("Matrix:", self.matrix,  sep="\n")
                print()

                start = time.perf_counter()
                calc = SimpleCalculation.matrix_rank(self.matrix)
                finish = time.perf_counter()

                self.time = round(finish - start, 3)
                print("Result:", calc)
                print()
                print("Time:", self.time)
                print()
                self.result_show(calc, 'rank')

        elif func == 'trace':
            self.matrix = np.zeros((a, a))

            try:
                for i in range(a):
                    for j in range(a):
                        if self.matrix_entries[i][j].get() == '':
                            self.matrix_entries[i][j].insert(0, '0')

                        self.matrix[i, j] = float(self.matrix_entries[i][j].get())
            except ValueError:
                is_error = True
                self.errors('alpha')
            except:
                is_error = True
                self.errors('unexpected')

            if not is_error:
                print("Matrix:", self.matrix, sep="\n")
                print()

                start = time.perf_counter()
                calc = SimpleCalculation.matrix_trace(self.matrix)
                finish = time.perf_counter()

                self.time = round(finish - start, 3)
                print("Result:", calc)
                print()
                print("Time:", self.time)
                print()
                self.result_show(calc, 'trace')

    def rand_calculate(self, func, a, b=0, c=0):
        """This method creates random matrices of given dimensions and proceeds
        with the desired calculation"""
        is_error = False

        if func == 'add_sub':
            try:
                a = int(a)
                b = int(b)
                if a < 2 or b < 2:
                    is_error = True
                    self.errors('dims')
            except ValueError:
                is_error = True
                self.errors('dims')

            if not is_error:
                try:
                    matrix_A = RandomMatrix.random_matrix(a, b)
                    matrix_B = RandomMatrix.random_matrix(a, b)

                    if self.rand_op.get() == '+':
                        start = time.perf_counter()
                        calc = SimpleCalculation.matrix_add(matrix_A, matrix_B)
                        finish = time.perf_counter()

                    elif self.rand_op.get() == '-':
                        start = time.perf_counter()
                        calc = SimpleCalculation.matrix_sub(matrix_A, matrix_B)
                        finish = time.perf_counter()
                except MemoryError:
                    is_error = True
                    self.errors('memory')
                except:
                    is_error = True
                    self.errors('unexpected')

                if not is_error:
                    print('Matrix A:', matrix_A, sep='\n')
                    print()
                    print('Matrix B:', matrix_B, sep='\n')
                    print()
                    self.time = round(finish - start, 3)
                    print('Result:', calc, sep='\n')
                    print()
                    print('Time:', self.time)
                    print()
                    self.rand_result_show(calc)

        elif func == 'mul_num':
            try:
                a = int(a)
                b = int(b)
                if a < 2 or b < 2:
                    is_error = True
                    self.errors('dims')
            except ValueError:
                is_error = True
                self.errors('dims')

            if not is_error:
                try:
                    matrix = RandomMatrix.random_matrix(a, b)

                    if self.rand_num_entry.get():
                        try:
                            num = float(self.rand_num_entry.get())
                        except ValueError:
                            is_error = True
                            self.errors('num')
                    else: num = 1

                    if not is_error:
                        start = time.perf_counter()
                        calc = SimpleCalculation.matrix_mul_num(matrix, num)
                        finish = time.perf_counter()
                except MemoryError:
                    is_error = True
                    self.errors('memory')
                except:
                    is_error = True
                    self.errors('unexpected')

                if not is_error:
                    print('Matrix:', matrix, sep='\n')
                    print()
                    self.time = round(finish - start, 3)
                    print('Result:', calc, sep='\n')
                    print()
                    print('Time:', self.time)
                    print()
                    self.rand_result_show(calc)

        elif func == 'mul':
            try:
                a = int(a)
                b = int(b)
                c = int(c)
                if a < 2 or b < 2 or c < 2:
                    is_error = True
                    self.errors('dims')
            except ValueError:
                is_error = True
                self.errors('dims')

            if not is_error:
                try:
                    matrix_A, matrix_B = RandomMatrix.two_random_matrices(a, b, c)

                    if min(a, b, c) > 1000:
                        start_mp = time.perf_counter()
                        calc = MultiprocessingCalculation.multiplication(matrix_A, matrix_B)
                        finish_mp = time.perf_counter()

                        self.time = round(finish_mp - start_mp, 3)
                        self.rand_result_show(calc)
                    else:
                        start_mp = time.perf_counter()
                        calc = MultiprocessingCalculation.multiplication(matrix_A, matrix_B)
                        finish_mp = time.perf_counter()

                        start_simple = time.perf_counter()
                        SimpleCalculation.matrix_mul(matrix_A, matrix_B)
                        finish_simple = time.perf_counter()
                except MemoryError:
                    is_error = True
                    self.errors('memory')
                except:
                    is_error = True
                    self.errors('unexpected')

                    self.time = round(min(finish_simple - start_simple, finish_mp - start_mp), 3)
                if not is_error:
                    print('Matrix A:', matrix_A, sep='\n')
                    print()
                    print('Matrix B:', matrix_B, sep='\n')
                    print()
                    self.time = round(min(finish_simple - start_simple, finish_mp - start_mp), 3)
                    print('Result:', calc, sep='\n')
                    print('Time w/o multiprocessing:', finish_simple - start_simple)
                    print('Tim w/ multiprocessing:', finish_mp - start_mp)
                    print()
                    self.rand_result_show(calc)

        elif func == 'power':
            try:
                a = int(a)
                if a < 2:
                    is_error = True
                    self.errors('dims')
            except ValueError:
                is_error = True
                self.errors('dims')

            if not is_error:
                try:
                    matrix = RandomMatrix.random_matrix(a, a)

                    if self.rand_power_entry.get():
                        if not self.rand_power_entry.get().isdigit():
                            is_error = True
                            self.errors('power')
                        else:
                            num = int(self.rand_power_entry.get())
                    else: num = 1
                    if not is_error:
                        start = time.perf_counter()
                        calc = SimpleCalculation.matrix_power(matrix, num)
                        finish = time.perf_counter()
                except MemoryError:
                    is_error = True
                    self.errors('memory')
                except:
                    is_error = True
                    self.errors('unexpected')

                if not is_error:
                    print('Matrix:', matrix, sep='\n')
                    print()
                    self.time = round(finish - start, 3)
                    print('Result:', calc, sep='\n')
                    print()
                    print('Time:', self.time)
                    print()
                    self.rand_result_show(calc)

        elif func == 'det':
            try:
                a = int(a)
                if a < 2:
                    is_error = True
                    self.errors('dims')
            except ValueError:
                is_error = True
                self.errors('dims')

            if not is_error:
                try:
                    matrix = RandomMatrix.random_matrix(a, a)

                    start = time.perf_counter()
                    calc = SimpleCalculation.matrix_det(matrix)
                    finish = time.perf_counter()
                except MemoryError:
                    is_error = True
                    self.errors('memory')
                except:
                    is_error = True
                    self.errors('unexpected')

            if not is_error:
                print('Matrix:', matrix, sep='\n')
                print()
                self.time = round(finish - start, 3)
                print('Result:', calc, sep='\n')
                print()
                print('Time:', self.time)
                print()
                self.rand_result_show(calc, 'det')

        elif func == 'inv':
            try:
                a = int(a)
                if a < 2:
                    is_error = True
                    self.errors('dims')
            except ValueError:
                is_error = True
                self.errors('dims')

            if not is_error:
                try:
                    matrix = RandomMatrix.random_matrix(a, a)

                    start = time.perf_counter()
                    calc = SimpleCalculation.matrix_inv(matrix)
                    finish = time.perf_counter()
                except MemoryError:
                    is_error = True
                    self.errors('memory')
                except:
                    is_error = True
                    self.errors('unexpected')

            if not is_error:
                print('Matrix:', matrix, sep='\n')
                print()
                self.time = round(finish - start, 3)
                print('Result:', calc, sep='\n')
                print()
                print('Time:', self.time)
                print()
                self.rand_result_show(calc)

        elif func == 'trans':
            try:
                a = int(a)
                b = int(b)
                if a < 2 or b < 2:
                    is_error = True
                    self.errors('dims')
            except ValueError:
                is_error = True
                self.errors('dims')

            if not is_error:
                try:
                    matrix = RandomMatrix.random_matrix(a, b)

                    start = time.perf_counter()
                    calc = SimpleCalculation.matrix_trans(matrix)
                    finish = time.perf_counter()
                except MemoryError:
                    is_error = True
                    self.errors('memory')
                except:
                    is_error = True
                    self.errors('unexpected')

                if not is_error:
                    print('Matrix:', matrix, sep='\n')
                    print()
                    self.time = round(finish - start, 3)
                    print('Result:', calc, sep='\n')
                    print()
                    print('Time:', self.time)
                    print()
                    self.rand_result_show(calc)

        elif func == 'rank':
            try:
                a = int(a)
                b = int(b)
                if a < 2 or b < 2:
                    is_error = True
                    self.errors('dims')
            except ValueError:
                is_error = True
                self.errors('dims')

            if not is_error:
                try:
                    matrix = RandomMatrix.random_matrix(a, b)

                    start = time.perf_counter()
                    calc = SimpleCalculation.matrix_rank(matrix)
                    finish = time.perf_counter()
                except MemoryError:
                    is_error = True
                    self.errors('memory')
                except:
                    is_error = True
                    self.errors('unexpected')

                if not is_error:
                    print('Matrix:', matrix, sep='\n')
                    print()
                    self.time = round(finish - start, 3)
                    print('Result:', calc, sep='\n')
                    print()
                    print('Time:', self.time)
                    print()
                    self.rand_result_show(calc, 'rank')

        elif func == 'trace':
            try:
                a = int(a)
                if a < 2:
                    is_error = True
                    self.errors('dims')
            except ValueError:
                is_error = True
                self.errors('dims')

            if not is_error:
                try:
                    matrix = RandomMatrix.random_matrix(a, a)

                    start = time.perf_counter()
                    calc = SimpleCalculation.matrix_trace(matrix)
                    finish = time.perf_counter()
                except MemoryError:
                    is_error = True
                    self.errors('memory')
                except:
                    is_error = True
                    self.errors('unexpected')

                if not is_error:
                    print('Matrix:', matrix, sep='\n')
                    print()
                    self.time = round(finish - start, 3)
                    print('Result:', calc, sep='\n')
                    print()
                    print('Time:', self.time)
                    print()
                    self.rand_result_show(calc, 'trace')

    def clear_cells(self, list):
        """This method clears all entry boxes"""
        for i in range(len(list)):
            for entry in list[i]:
                entry.delete(0, 'end')

    def fill_zeros(self, list):
        """This method fills all empty entry boxes with the number '0'"""
        for i in range(len(list)):
            for entry in list[i]:
                if entry.get() == '':
                    entry.insert(0, '0')

    def fill_ones(self, list):
        """This method fills all empty entry boxes with the number '1'"""
        for i in range(len(list)):
            for entry in list[i]:
                if entry.get() == '':
                    entry.insert(0, '1')

    def mem_sv(self, list):
        """This method saves the inputted matrix in the memory for future use"""
        is_error = False  # this variable checks if there is error(element isaplha)

        self.matrix_saved = np.zeros((len(list), len(list[0])))
        try:
            self.matrix_saved = np.zeros((len(list), len(list[0])))
            for i in range(len(list)):
                for j in range(len(list[0])):
                    self.matrix_saved[i][j] = list[i][j].get()
        except ValueError:
            is_error = True
            self.errors('mem_save')
        except:
            is_error = True
            self.errors('unexpected')

        if not is_error:
            print("Matrix in memory:", self.matrix_saved, sep="\n")
            print()

    def mem_ld(self, list):
        """This method loads the matrix saved in the memory into the entry boxes"""
        try:
            if self.matrix_saved.shape[0] != len(list) or self.matrix_saved.shape[1] != len(list[0]):
                self.errors('mem_load_dims')
            else:
                GUI.clear_cells(self, list)
                for i in range(len(list)):
                    for j in range(len(list[0])):
                        list[i][j].insert(0, self.matrix_saved[i][j])
        except AttributeError:
            self.errors('mem_load_empty')


    def result_show(self, result, func=''):
        """This method creates a new window displaying the result"""
        self.result_win = Toplevel(root, bg=self.color_bg1)
        self.result_win.iconbitmap('matrix_ico.ico')
        self.result_win.title('Matrix Calculator')
        self.result_win.geometry("600x400")

        self.f_res_text = Frame(self.result_win, bg=self.color_bg1)
        self.f_res_text.pack(side='top', fill='x', padx=40, pady=(40, 20))

        self.f_main_res = Frame(self.result_win, bg=self.color_bg1, padx=5, pady=5)
        self.f_main_res.pack(side='top', padx=40)

        self.f_time = Frame(self.result_win, bg=self.color_bg1)
        self.f_time.pack(side='bottom', fill='x', padx=40, pady=(10, 40))

        self.time_text = Label(self.f_time, text=f'Computation time: {self.time} seconds', font=('Arial', 10),
                               bg=self.color_bg1, fg=self.color_text2)
        self.time_text.pack(anchor='e')

        self.result_text = Label(self.f_res_text, text='Result:', font=('Arial', 12), bg=self.color_bg1,
                                 fg=self.color_text2)
        self.result_text.pack(anchor='w')

        labels = []
        width = 3

        if func == 'det':
            self.result_win.geometry("600x300")

            self.result = Label(self.f_main_res, text=f'Matrix determinant is {round(result, 2)}', font=('Arial', 15),
                                bg=self.color_bg1, fg=self.color_text2)
            self.result.pack(anchor='n')

        elif func == 'rank':
            self.result_win.geometry("600x300")

            self.result = Label(self.f_main_res, text=f'Matrix rank is {round(result, 2)}', font=('Arial', 15),
                                bg=self.color_bg1, fg=self.color_text2)
            self.result.pack(anchor='n')

        elif func == 'trace':
            self.result_win.geometry("600x300")

            self.result = Label(self.f_main_res, text=f'Matrix trace is {round(result, 2)}', font=('Arial', 15),
                                bg=self.color_bg1, fg=self.color_text2)
            self.result.pack(anchor='n')

        else:
            self.f_result = Frame(self.f_main_res, bg=self.color_bg2)
            self.f_result.pack(anchor='w')

            self.result_win.geometry("830x550")

            Label(self.f_result, text=' ', font=('Arial', 12), bg=self.color_bg1, fg='grey',
                  width=2).grid(row=0, column=0, padx=2, pady=2)

            for j in range(len(result[0])):
                label = Label(self.f_result, text='A'+f'{j + 1}', font=('Arial', 12), bg=self.color_bg1, fg='grey')
                label.grid(row=0, column=j+1, padx=2)
                labels.append(label)

            for i in range(len(result)):
                Label(self.f_result, text=i + 1, font=('Arial', 12), bg=self.color_bg1, fg='grey',
                      width=2).grid(row=i+1, column=0, padx=2)
                for j in range(len(result[0])):

                    if result[i][j].is_integer():
                        number = int(result[i][j])
                    else:
                        number = result[i][j]

                    if len(str(number)) > width:
                        width = len(str(number))

                    label = Label(self.f_result, text=number, font=('Arial', 12), bg=self.color_bg1,
                                  fg=self.color_text2)
                    label.grid(row=i + 1, column=j + 1, padx=2, pady=2)
                    labels.append(label)

        for label in labels: label.configure(width=width)

    def rand_result_show(self, result, func=''):
        """This method creates a new window displaying the result (from the
        random matrices)"""
        self.result_win = Toplevel(root, bg=self.color_bg1)
        self.result_win.iconbitmap('matrix_ico.ico')
        self.result_win.title('Matrix Calculator')

        self.f_res_text = Frame(self.result_win, bg=self.color_bg1)
        self.f_res_text.pack(side='top', fill='x', padx=40, pady=(40, 20))

        self.f_main_res = Frame(self.result_win, bg=self.color_bg1, padx=5, pady=5)
        self.f_main_res.pack(side='top', padx=40)

        self.f_time = Frame(self.result_win, bg=self.color_bg1)
        self.f_time.pack(side='bottom', fill='x', padx=40, pady=(10, 40))

        self.time_text = Label(self.f_time, text=f'Computation time: {self.time} seconds', font=('Arial', 10),
                               bg=self.color_bg1, fg=self.color_text2)
        self.time_text.pack(anchor='e')

        self.result_text = Label(self.f_res_text, text='Result:', font=('Arial', 12), bg=self.color_bg1,
                                 fg=self.color_text2)
        self.result_text.pack(anchor='w')

        if func == 'det':
            self.result_win.geometry("600x300")

            self.result = Label(self.f_main_res, text=f'Matrix determinant is {round(result, 2)}', font=('Arial', 15),
                                 bg=self.color_bg1, fg=self.color_text2)
            self.result.pack(anchor='n')

        elif func == 'rank':
            self.result_win.geometry("600x300")

            self.result = Label(self.f_main_res, text=f'Matrix rank is {round(result, 2)}', font=('Arial', 15),
                                bg=self.color_bg1, fg=self.color_text2)
            self.result.pack(anchor='n')

        elif func == 'trace':
            self.result_win.geometry("600x300")

            self.result = Label(self.f_main_res, text=f'Matrix trace is {round(result, 2)}', font=('Arial', 15),
                                bg=self.color_bg1, fg=self.color_text2)
            self.result.pack(anchor='n')

        else:
            self.result_win.geometry("830x550")

            self.scroll_x = Scrollbar(self.f_main_res, orient="horizontal")
            self.scroll_y = Scrollbar(self.f_main_res, orient="vertical")
            self.scroll_x.pack(side='bottom', fill='x')
            self.scroll_y.pack(side='right', fill='y')

            self.t_result = Text(self.f_main_res, font=('Arial', 12), bg=self.color_bg2, fg=self.color_text2,
                                 spacing1=10, height=12, width=80, relief=GROOVE,
                                 xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set, wrap='none')
            self.t_result['font'] = ('Arial', 12)
            self.t_result.pack(side='left')

            if len(str(np.amax(result))) < 7: tabs = 1
            elif len(str(np.amax(result))) < 12: tabs = 2
            elif len(str(np.amax(result))) < 17: tabs = 3
            elif len(str(np.amax(result))) < 22: tabs = 4
            elif len(str(np.amax(result))) < 27: tabs = 5
            else: tabs = 6

            for i in range(len(result)):
                for j in range(len(result[0])):
                    self.t_result.insert('end', result[i][j])
                    if j == len(result[0])-1:
                        self.t_result.insert('end', '\n')
                    else:
                        self.t_result.insert('end', '\t'*tabs)

            self.scroll_x.config(command=self.t_result.xview)
            self.scroll_y.config(command=self.t_result.yview)

    def errors(self, type):
        if type == 'alpha':
            mb.showerror(title='Error', message="Array's elements must be numbers")
        elif type == 'num':
            mb.showerror(title='Error', message='Multiplying number input must be a number')
        elif type == 'unexpected':
            mb.showerror(title='Error', message='An unexpected error has occurred')
        elif type == 'power':
            mb.showerror(title='Error', message='Power number input must be a positive integer')
        elif type == 'memory':
            mb.showerror(title='Error', message='Input values are too big. Try smaller values')
        elif type == 'dims':
            mb.showerror(title='Error', message='Dimension inputs must be integers larger or equal than 2')
        elif type == 'singular':
            mb.showerror(title='Error', message='Matrix is singular thus is not invertible')
        elif type == 'mem_save':
            mb.showerror(title='Error', message="Array's elements in saved matrix must be numbers")
        elif type == 'mem_load_empty':
            mb.showerror(title='Error', message='No matrix in memory')
        elif type == 'mem_load_dims':
            mb.showerror(title='Error', message='Matrix in memory does not match current dimensions')


if __name__ == '__main__':
    root = Tk()
    gui = GUI(root)
    root.mainloop()
