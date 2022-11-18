from operator import abs
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt, pyplot
from tkinter import *
from tkinter.ttk import *


form = Tk()
classes = ['Adelie', 'Gentoo', 'Chinstrap']
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g']
data1 = StringVar()
data2 = StringVar()
data3 = StringVar()
data4 = StringVar()
label1 = StringVar()
label2 = StringVar()
label3 = StringVar()
label4 = StringVar()
label5 = StringVar()
var1 = DoubleVar()
var2 = IntVar()
var3 = DoubleVar()
radio_var = IntVar()
#combobox
class1 = Combobox(form, width=20, textvariable=data1)
class2 = Combobox(form, width=20, textvariable=data2)
feature1 = Combobox(form, width=20, textvariable=data3)
feature2 = Combobox(form, width=20, textvariable=data4)
selectedClass1 = ''
selectedClass2 = ''
selectedFeature1 = ''
selectedFeature2 = ''
lR = 0
epoch_num = 0
mse_threshold = 0
use_bias = True
train_data = []
train_labels = []
test_data = []
test_labels = []
weights = []
encodes = {}


# take user values
def user_inputs():
    global selectedClass1, selectedClass2, selectedFeature1, selectedFeature2, lR, epoch_num, use_bias, mse_threshold
    selectedClass1 = data1.get()
    selectedClass2 = data2.get()
    selectedFeature1 = data3.get()
    selectedFeature2 = data4.get()
    lR = var1.get()
    epoch_num = var2.get()
    mse_threshold = var3.get()
    use_bias = True
    if radio_var.get() == 1:
        use_bias = True
    elif radio_var.get() == 2:
        use_bias = False


def initialize_Model_Dfs():
    user_inputs()
    global train_data, train_labels, test_data, test_labels, weights, encodes
    # create train & test data based on user selection
    # 1) select species
    train_frames = []
    test_frames = []
    if selectedClass1 == 'Adelie':
        train_frames.append(Adelie_train)
        test_frames.append(Adelie_test)
    elif selectedClass1 == 'Gentoo':
        train_frames.append(Gentoo_train)
        test_frames.append(Gentoo_test)
    else:
        train_frames.append(Chinstrap_train)
        test_frames.append(Chinstrap_test)

    if selectedClass2 == 'Adelie':
        train_frames.append(Adelie_train)
        test_frames.append(Adelie_test)
    elif selectedClass2 == 'Gentoo':
        train_frames.append(Gentoo_train)
        test_frames.append(Gentoo_test)
    else:
        train_frames.append(Chinstrap_train)
        test_frames.append(Chinstrap_test)

    train_data = pd.concat(train_frames).reset_index(drop=True)
    test_data = pd.concat(test_frames).reset_index(drop=True)

    # 2) keep only selected features
    train_data = train_data[['species', selectedFeature1, selectedFeature2]]
    test_data = test_data[['species', selectedFeature1, selectedFeature2]]

    # encode species column
    encodes = {1: selectedClass1, -1: selectedClass2}
    train_data['species'].replace(to_replace=[selectedClass1, selectedClass2], value=['1', '-1'], inplace=True)
    test_data['species'].replace(to_replace=[selectedClass1, selectedClass2], value=['1', '-1'], inplace=True)
    # data shuffling
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    # separate labels
    train_labels = train_data.pop('species')
    test_labels = test_data.pop('species')

    # weight & bias

    if use_bias:
        weights = np.random.rand(3)
    else:
        weights = np.random.rand(2)


# single_layer_Perceptron function (train function)
def run_single_layer():
    global weights, train_labels, train_data, epoch_num

    # convert data frame to numpy
    trainData = train_data.to_numpy()
    trainData = preprocessing.normalize(trainData)
    trainLabel = train_labels
    # transpose weight list for dot product
    transpose_weight = weights.transpose()
    bias = 1
    start_epoch_num = epoch_num
    while epoch_num:
        row_num = 0
        score = 0
        for row in trainData:
            # in case bias add bias value in feature list
            if len(weights) > 2:
                row = np.append(row, bias)
            predictedValue = np.dot(row, transpose_weight)
            error = int(trainLabel[row_num]) - float(predictedValue)

            # if error occurs call update_weight function to update weights
            if error != 0:
                weights = update_weight(transpose_weight, row, error)

            if signum(predictedValue) == int(trainLabel[row_num]):
                score += 1
            row_num += 1
            # print train accuracy
        print('epoch ',  start_epoch_num - epoch_num, 'accuracy :', (score / 60.0) * 100)
        epoch_num -= 1

        if calcLMS(trainData) < mse_threshold:
            break


def calcLMS(trainData):
    transpose_weight = weights.transpose()
    errors = []
    error_summation = 0.0
    row_num = 0
    for row in trainData:
        # in case bias add bias value in feature list
        if len(weights) > 2:
            row = np.append(row, 1)
        predictedValue = np.dot(transpose_weight, row)
        error = int(train_labels[row_num]) - float(predictedValue)
        errors.append(error)
        row_num += 1

    for error in errors:
        error_summation += (error * error)
    error_summation /= 2.0
    error_summation /= 60.0
    return error_summation


# update weights when an error occurs (call by single_layer_Perceptron function)
def update_weight(weight_matrix, row, error_value):
    global lR
    delta = lR * error_value * row
    weight_matrix += delta
    return weight_matrix


# test sample
def testSample(sample):
    global weights, encodes
    # add bias if exist
    if len(weights) > 2:
        sample = np.append(sample, 1)
    transpose_weight = weights.transpose()

    # calculate predicted y
    predictedValue_y = np.dot(transpose_weight, sample)
    predictedValue_y = signum(predictedValue_y)
    print("the ClassID :", encodes[predictedValue_y])


# test data, print accuracy and confusionMatrix
def test():
    global test_labels, test_data, weights
    testData = test_data.to_numpy()
    testData = preprocessing.normalize(testData)
    transpose_weight = weights.transpose()
    test_label = test_labels
    row_num = 0
    x0 = 1
    score = 0
    confusionMatrix = {'Class1T': 0, 'Class1F': 0, 'Class2T': 0, 'Class2F': 0}
    for row in testData:
        # add bias if exist
        if len(weights) > 2:
            row = np.append(row, x0)

        # calculate predicted y
        predictedValue = np.dot(row, transpose_weight)

        # updated confusion matrix
        # if false prediction
        if signum(predictedValue) == int(test_labels[row_num]):
        # if sample was from class 1
            if test_labels[row_num] == '1':
                confusionMatrix['Class1T'] += 1
            # if sample was from class 2
            else:
                confusionMatrix['Class2T'] += 1
            score = score + 1
        # if correct prediction
        else:
            if predictedValue == '1':
                confusionMatrix['Class2F'] += 1
            else:
                confusionMatrix['Class1F'] += 1
        row_num += 1
    accuracy = (score / 40.0) * 100
    print("accuracy:", accuracy, "and the score: ", score)
    print("confusion Matrix : ", confusionMatrix)


def decision_boundry():
    # find X & Y for line
    # X1 & X2 are min and max value in feature 1
    min_feature1 = min(test_data[selectedFeature1])
    max_feature1 = max(test_data[selectedFeature1])

    # calculate Y1& Y2 from line eq: W0X1 + W1X2 +W2 = 0
    # if there is bias
    if use_bias:
        y1 = ((weights[2] * -1) - (min_feature1 * weights[0])) / weights[1]
        y2 = ((weights[2] * -1) - (max_feature1 * weights[0])) / weights[1]
    else:
        y1 = -(min_feature1 * weights[0]) / weights[1]
        y2 = -(max_feature1 * weights[0]) / weights[1]

    x = [min_feature1, max_feature1]
    y = [y1, y2]

    # create the figure
    figureName = 'decision_boundry'
    plt.figure(figureName)

    # plot training data as a scatter
    if selectedClass1 == 'Adelie' or selectedClass2 == 'Adelie':
        plt.scatter(Adelie_train[selectedFeature1], Adelie_train[selectedFeature2])
    if selectedClass1 == 'Gentoo' or selectedClass2 == 'Gentoo':
        plt.scatter(Gentoo_train[selectedFeature1], Gentoo_train[selectedFeature2])
    if selectedClass1 == 'Chinstrap' or selectedClass2 == 'Chinstrap':
        plt.scatter(Chinstrap_train[selectedFeature1], Chinstrap_train[selectedFeature2])

    # add x & y axes labels
    plt.xlabel(selectedFeature1)
    plt.ylabel(selectedFeature2)

    # add plot legend for classes
    plt.legend((selectedClass1, selectedClass2),
               scatterpoints=1,
               fontsize=8
               )
    plt.plot(x, y)
    plt.show()


# call back when press the run button
def run():
    initialize_Model_Dfs()
    run_single_layer()
    test()
    # bill_depth_mm & flipper_length_mm for gentoo sample
    sample = [42, 13.5]
    testSample(np.array(sample))
    decision_boundry()


# signum activation function
def signum(num):
    if num > 0:
        return 1
    else:
        return -1


# create labels in gui
def create_label():
    class_label = Label(form, textvariable=label1)
    label1.set("Select the two species")
    class_label.place(x=20, y=30)

    feature_label = Label(form, textvariable=label2)
    label2.set("Select the two features")
    feature_label.place(x=20, y=120)

    lr_label = Label(form, textvariable=label3)
    label3.set("learning rate")
    lr_label.place(x=20, y=220)

    mse_label = Label(form, textvariable=label4)
    label4.set("mse_threshold")
    mse_label.place(x=20, y=255)

    epoch_label = Label(form, textvariable=label5)
    label5.set("epochs number")
    epoch_label.place(x=250, y=220)


# create Radio buttons in gui
def create_radio():
    r1 = Radiobutton(form, text="bias", width=120, variable=radio_var, value=1)
    r1.pack(anchor=W)
    r1.place(x=120, y=290)

    r2 = Radiobutton(form, text="no bias", width=120, variable=radio_var, value=2)
    r2.pack(anchor=W)
    r2.place(x=300, y=290)


# create  run_button in gui
def create_button():
    btn = Button(form, text="Run", command=run)
    btn.place(x=190, y=350)


# create spinbox in gui
def create_spinbox():
    spin1 = Spinbox(form, from_=0, to=1, increment=0.1, width=5, textvariable=var1)
    spin1.place(x=120, y=220)

    spin2 = Spinbox(form, from_=1, to=5000, increment=10, width=5, textvariable=var2)
    spin2.place(x=350, y=220)

    spin3 = Spinbox(form, from_=0, to=100, increment=0.5, width=5, textvariable=var3)
    spin3.place(x=120, y=255)


# create combobox in gui
def create_combo():
    class1['values'] = classes
    class1.grid(column=1, row=3)
    class1.place(x=20, y=60)
    class1.current()
    data1.trace('w', update)

    class2['values'] = classes
    class2.grid(column=1, row=2)
    class2.place(x=260, y=60)
    class2.current()

    feature1['values'] = features
    feature1.grid(column=1, row=5)
    feature1.place(x=20, y=150)
    feature1.current()
    data3.trace('w', update2)

    feature2['values'] = features
    feature2.grid(column=1, row=4)
    feature2.place(x=260, y=150)
    feature2.current()


# fill class2 Combobox by all data except the chosen class in another Combobox
def update(*args):
    data2.set('')
    class2["values"] = [x for x in classes if x != data1.get()]


# fill feature2 Combobox by all data except the chosen feature in another Combobox
def update2(*args):
    data4.set('')
    feature2["values"] = [x for x in features if x != data3.get()]


# call all function that create elements in Gui
def gui():
    form.geometry("450x450")
    form.title("Form")
    create_label()
    create_combo()
    create_spinbox()
    create_radio()
    create_button()
    form.mainloop()


def data_preprocessing():
    dataSet = pd.read_csv('penguins.csv')

    # find important columns name which contain  numeric values
    numbers_cols = dataSet.select_dtypes(include=np.number).columns.to_list()

    # find important columns name which contain nun numeric values & convert it's type to string
    non_integer_cols = dataSet.select_dtypes(include=['object']).columns.to_list()
    dataSet[non_integer_cols] = dataSet[non_integer_cols].astype('string')

    # split dataSet based on specie
    adelie = dataSet.iloc[0:50, :]
    gentoo = dataSet.iloc[50: 100, :]
    chinstrap = dataSet.iloc[100: 150, :]

    nan_val_in_Adelie = {}
    nan_val_in_Gentoo = {}
    nan_val_in_Chinstrap = {}

    # find values for 'nan' with median in integer cols & with most repeated value in 'gender' col.
    # for integer col
    for col in numbers_cols:
        nan_val_in_Adelie[col] = adelie[col].median()
        nan_val_in_Gentoo[col] = gentoo[col].median()
        nan_val_in_Chinstrap[col] = chinstrap[col].median()

    # for gender
    nan_val_in_Adelie['gender'] = adelie['gender'].mode()[0]
    nan_val_in_Gentoo['gender'] = gentoo['gender'].mode()[0]
    nan_val_in_Chinstrap['gender'] = chinstrap['gender'].mode()[0]

    # replace nan
    # in adelie
    adelie = adelie.fillna(value=nan_val_in_Adelie)
    # in gentoo
    gentoo = gentoo.fillna(value=nan_val_in_Gentoo)
    # in Chinstrap
    chinstrap = chinstrap.fillna(value=nan_val_in_Chinstrap)

    # Encoding gender column
    genders = ['male', 'female']
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(genders)
    adelie[adelie.columns[4]] = label_encoder.transform(adelie['gender'])
    gentoo[gentoo.columns[4]] = label_encoder.transform(gentoo['gender'])
    chinstrap[chinstrap.columns[4]] = label_encoder.transform(chinstrap['gender'])

    # dataSet shuffling
    adelie = adelie.sample(frac=1).reset_index(drop=True)
    gentoo = gentoo.sample(frac=1).reset_index(drop=True)
    chinstrap = chinstrap.sample(frac=1).reset_index(drop=True)

    # split dataSet into train dataSet and test dataSet
    Adelie_train = adelie.iloc[:30, :]
    Adelie_test = adelie.iloc[30:, :].reset_index(drop=True)
    Gentoo_train = gentoo.iloc[:30, :]
    Gentoo_test = gentoo.iloc[30:, :].reset_index(drop=True)
    Chinstrap_train = chinstrap.iloc[:30, :]
    Chinstrap_test = chinstrap.iloc[30:, :].reset_index(drop=True)

    return Adelie_train, Adelie_test, Gentoo_train, Gentoo_test, Chinstrap_train, Chinstrap_test


Adelie_train, Adelie_test, Gentoo_train, Gentoo_test, Chinstrap_train, Chinstrap_test = data_preprocessing()

gui()



