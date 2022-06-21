# coding=UTF-8
import numpy as np
import math
import matplotlib.pyplot as plt
def simulate_data(shape):
    output_array = np.zeros(shape=shape, dtype='float32')
    sum_array1 = output_array * 1
    sum_array2 = output_array * 1
    for i in range(output_array.shape[0]):
        for j in range(output_array.shape[1]):
            sum_array1[i, j] = np.exp(-1 * ((i - 20) ** 2 + (j - 20) ** 2) / (2 * 30 ** 2))
            sum_array2[i, j] = -1 * np.exp(-1 * ((i - 80) ** 2 + (j - 80) ** 2) / (2 * 30 ** 2))
    output_array = output_array + sum_array1 + sum_array2
    np.save(r"simulate_array.npy", output_array)
    return

def seed_point(shape, seed_number):
    seed_point_list = []
    # i = 0
    # while i < seed_number:
    #     check_distance = 0
    #     new_seed = (np.random.randint(shape[0]), np.random.randint(shape[1]))
    #     if i == 0:
    #         seed_point_list.append(new_seed)
    #         i += 1
    #     else:
    #         for j in range(len(seed_point_list)):
    #             if (new_seed[0] - seed_point_list[j][0]) ** 2 + (new_seed[1] - seed_point_list[j][1]) ** 2 < 25:
    #                 check_distance += 1
    #                 break
    #         if check_distance == 0:
    #             seed_point_list.append(new_seed)
    #             i += 1
    # b = np.array(seed_point_list)
    for i in range(3):
        for j in range(5):
            seed_point_list.append((i*30+10, j*20+10))
    b = np.array(seed_point_list)
    np.save(r"seed_point.npy", b)
    return

def distance_half(simulate_data_file, seed_point_file):
    simulate_array = np.load(simulate_data_file)
    seed_point_array = np.load(seed_point_file)
    print('simulate_array shape', simulate_array.shape)
    print('seed_point shape', seed_point_array.shape)
    distance_array = np.zeros(shape=(seed_point_array.shape[0], seed_point_array.shape[0]), dtype='float32')
    variance_array = np.zeros(shape=(seed_point_array.shape[0], seed_point_array.shape[0]), dtype='float32')
    for i in range(seed_point_array.shape[0]):
        for j in range(seed_point_array.shape[0]):
            distance_array[i, j] = ((seed_point_array[i][0]-seed_point_array[j][0])**2+(seed_point_array[i][1]-seed_point_array[j][1])**2)**0.5
            variance_array[i, j] = (simulate_array[seed_point_array[i][0], seed_point_array[i][1]]-simulate_array[seed_point_array[j][0], seed_point_array[j][1]])**2/2
    plt.subplot(1, 2, 1)
    plt.imshow(distance_array, cmap=plt.cm.rainbow)
    plt.subplot(1, 2, 2)
    plt.imshow(variance_array, cmap=plt.cm.rainbow)
    plt.show()
    np.savetxt('variance_matrix.csv', variance_array, delimiter=',')
    # distance_array = distance_array.reshape(seed_point_array.shape[0]**2,)
    # variance_array = variance_array.reshape(seed_point_array.shape[0]**2,)
    # output = np.vstack([distance_array, variance_array])
    # np.savetxt('distance_variance.csv', np.moveaxis(output, 0, -1), delimiter=',')
    return
def distance_variance_predict(distance_variance_file):
    seed_point_number = np.zeros(shape=(10,), dtype='float')
    distance = seed_point_number * 1
    variance = seed_point_number * 1
    distance_variance_array = np.loadtxt(distance_variance_file, delimiter=',')
    for i in range(distance_variance_array.shape[0]):
        index = int(distance_variance_array[i, 0]/10)
        if 0<=index<10:
            seed_point_number[index] += 1
            distance[index] += distance_variance_array[i, 0]
            variance[index] += distance_variance_array[i, 1]
    distance_mean = distance/seed_point_number
    variance_mean = variance/seed_point_number
    output = np.vstack([distance_mean, variance_mean])
    np.savetxt('mean_distance_variance.csv', np.moveaxis(output, 0, -1), delimiter=',')
    return
def inverse_matrix_calculate():
    origin_matrix = np.ones(shape=(16, 16), dtype='float32')
    input_array = np.loadtxt(r"variance_matrix.csv", delimiter=',', dtype='float32')
    origin_matrix[:15, :15] = input_array
    origin_matrix[15, 15] = 0
    origin_matrix = np.mat(origin_matrix, dtype='float32')
    inverse_matrix = origin_matrix.getI()
    aba = inverse_matrix * origin_matrix
    np.savetxt('chech_origin_matrix.csv', origin_matrix, delimiter=',')
    np.savetxt('check_inverse.csv', aba, delimiter=',')
    np.savetxt('inverse_matrix.csv', inverse_matrix, delimiter=',')
    return inverse_matrix
def interplote_weight(index_turple, inverse_matrix):
    seed_point_array = np.load(r"seed_point.npy")
    distance_array = np.zeros(shape=(15, ), dtype='float')
    variance_array = distance_array * 1
    for i in range(15):
        distance_array[i] = ((index_turple[0] - seed_point_array[i][0]) ** 2 + (index_turple[1] - seed_point_array[i][1]) ** 2) ** 0.5
    # print('distance_array:', distance_array)
    for i in range(15):
        if 0 <= distance_array[i] < 15:
            variance_array[i] = 0.0494/15*distance_array[i]
        elif 15 <= distance_array[i] < 93:
            variance_array[i] = 0.00002 * distance_array[i] ** 2 + 0.0083 * distance_array[i] - 0.0794
        else:
            variance_array[i] = 0.00002 * 93 ** 2 + 0.0083 * 93 - 0.0794
    # print ('variance_array:', variance_array)
    variance_vector = np.ones(shape=(16, ), dtype='float')
    variance_vector[:15] = variance_array
    variance_vector = np.mat(variance_vector).T
    # print ('variance_vector:', variance_vector)
    # print('inverse_matrix:', inverse_matrix)
    output = inverse_matrix * variance_vector
    output = output[:15, 0]
    sum_output = 0
    for i in range(15):
        a = np.abs(output[i, 0])
        if a > sum_output:
            sum_output = a
    output = output/sum_output
    print('output:', output)
    e_x = np.exp(output)
    output = e_x/np.sum(e_x)
    return output


def krige_interplote():
    simulate_array = np.load(r"simulate_array.npy")
    seed_point_array = np.load(r"seed_point.npy")
    # seed_point_array = seed_point_array[:2, :]
    print('seed_point_array:', seed_point_array)
    parameter_vector = np.zeros(shape=(15, ), dtype='float')
    inverse_matrix = inverse_matrix_calculate()
    for i in range(2):
        print(seed_point_array[i][0], seed_point_array[i][1])
        parameter_vector[i] = simulate_array[seed_point_array[i][0], seed_point_array[i][1]]
    parameter_vector = np.mat(parameter_vector)
    interplate_result = np.zeros(shape=(80, 100), dtype='float')
    # print('parameter:', parameter_vector)
    # weight = interplote_weight((0, 0), inverse_matrix)
    # print('weight:', weight)
    # interplate_result[0, 0] = parameter_vector * weight[:20, 0]
    for i in range(80):
        for j in range(100):
            weight = interplote_weight((i, j), inverse_matrix)
            # print('weight:', weight, weight.shape)
            # print('parameter_vector:', parameter_vector, parameter_vector.shape)
            interplate_result[i, j] = parameter_vector * weight
    plt.subplot(1, 2, 1)
    plt.imshow(simulate_array, cmap=plt.cm.rainbow)
    plt.scatter(seed_point_array[:, 1], seed_point_array[:, 0])
    plt.subplot(1, 2, 2)
    plt.imshow(interplate_result, cmap=plt.cm.rainbow)
    plt.show()
    return
if __name__ == '__main__':
    print('hello!')
    # simulate_data((80, 100))
    # plt.imshow(simulate_array, cmap=plt.cm.rainbow)
    # plt.show()
    # seed_point((80, 100), 20)
    # simulate_array = np.load(r"simulate_array.npy")
    # print('simulate_array shape:', simulate_array.shape)
    # seed_point = np.load(r"seed_point.npy")
    # x = seed_point[:, 0]
    # y = seed_point[:, 1]
    # print('x, y', x, y)
    # plt.imshow(simulate_array, cmap=plt.cm.rainbow)
    # plt.scatter(y, x)
    # plt.show()
    # distance_half(r"simulate_array.npy", "seed_point.npy")
    # distance_variance_predict('distance_variance.csv')
    # inverse_matrix_calculate()
    # interplote_weight((10, 20), 0)
    krige_interplote()
    # a = np.array([1, 2, 4, 8]).reshape(2,2)
    # print(a)
    # b = np.mat(a)
    # c = b.I
    # print(c)
    # d = b * c
    # print(d)