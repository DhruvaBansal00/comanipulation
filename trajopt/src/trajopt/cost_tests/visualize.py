import matplotlib.pyplot as plt

def histogram():
    file_name = './visibility.txt'
    f = open(file_name, 'r')
    lines = f.readlines()
    nums = []
    for x in lines:
        nums.append(float(x.strip('\n')))
    intervals = [(0 + i)/100.0 for i in range(0,100,5)]
    intervals.append(0.99)
    plt.hist(nums, bins=intervals)
    plt.show()

histogram()