import math
import sys

import numpy as np
from scipy import stats

__TRAIN_FILE = 'adult.data'
__TEST_FILE = 'adult.test'
__ATTRIBUTE = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'
]
__CONTINUOUS_ATTRIBUTE = [
    'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
    'hours_per_week'
]
__LABEL = ['<=50K', '>50K']


def get_samples():
    total_sample_num = 0

    sample = {
        label: {
            att: [] if att in __CONTINUOUS_ATTRIBUTE else {}
            for att in __ATTRIBUTE
        }
        for label in __LABEL
    }

    sample_num = {label: 0 for label in __LABEL}

    with open(__TRAIN_FILE) as train_file:
        # iterate over target frames
        for line_idx, line in enumerate(train_file.readlines()):
            values = line[:-1].split(', ')  # omitting incomplete data lines
            if '?' in values:
                continue
            else:
                # age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, label = values
                label = values[-1]

                for att_idx, att_value in enumerate(values[:-1]):
                    att_name = __ATTRIBUTE[att_idx]
                    # append continuous attributes
                    if att_name in __CONTINUOUS_ATTRIBUTE:
                        sample[label][att_name].append(int(att_value))
                    # update discrete attributes
                    else:
                        if att_value in sample[label][att_name]:
                            sample[label][att_name][att_value] += 1
                        else:
                            sample[label][att_name][att_value] = 1

                sample_num[label] += 1

                total_sample_num += 1

    return sample, sample_num, total_sample_num


def get_prior(sample_num, total_sample_num):
    p_50k_lower = sample_num['<=50K'] / total_sample_num
    p_50k_higher = sample_num['>50K'] / total_sample_num

    return p_50k_lower, p_50k_higher


def get_alpha(sample, sample_num):
    alpha = {
        label: {
            att: {} if att in __CONTINUOUS_ATTRIBUTE else
            {_class: .0
             for _class in sample[label][att].keys()}
            for att in __ATTRIBUTE
        }
        for label in __LABEL
    }

    for label in __LABEL:
        for att in __ATTRIBUTE:
            if att in __CONTINUOUS_ATTRIBUTE:
                alpha[label][att]['mean'] = np.mean(sample[label][att])
                alpha[label][att]['var'] = np.var(sample[label][att])
            else:
                class_condition_num = 0
                for _class in alpha[label][att].keys():
                    class_condition_num += sample[label][att][_class]
                for _class in alpha[label][att].keys():
                    alpha[label][att][_class] = sample[label][att][
                        _class] / class_condition_num

    return alpha


def test_samples(alpha, prior, test_line_num):
    result = {
        line_idx: {label: .0
                   for label in __LABEL}
        for line_idx in range(1, test_line_num + 1)
    }
    with open(__TRAIN_FILE) as train_file:
        # iterate over target frames
        for line_idx, line in enumerate(train_file.readlines()):
            if line_idx < test_line_num:
                values = line[:-1].split(', ')
                # omitting incomplete data lines
                if '?' in values:
                    continue
                else:
                    for label in __LABEL:
                        sample_prob = prior[label]
                        for att_idx, att_value in enumerate(values[:-1]):
                            att_name = __ATTRIBUTE[att_idx]
                            if att_name in __CONTINUOUS_ATTRIBUTE:
                                mu = alpha[label][att_name]['mean']
                                sigma = math.sqrt(
                                    alpha[label][att_name]['var'])
                                prob = stats.norm.pdf(
                                    int(att_value), mu, sigma)
                                # print(att_name, att_value, prob)
                                # sys.exit(0)
                            else:
                                prob = alpha[label][att_name][att_value]
                            log_prob = math.log(prob)
                            sample_prob += log_prob
                        result[line_idx + 1][label] = sample_prob
    return result


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


if __name__ == "__main__":
    sample, sample_num, total_sample_num = get_samples()

    # Quesion 5.1 (a)
    p_50k_lower, p_50k_higher = get_prior(sample_num, total_sample_num)
    print(
        'Quesion 5.1 (a)\nPrior probability of each class:\nP(<=50K)={}\nP(>50K)={}\n'
        .format(round(p_50k_lower, 4), round(p_50k_higher, 4)))

    # Quesion 5.1 (b)
    alpha = get_alpha(sample, sample_num)
    print('Quesion 5.1 (b)')
    for label_idx, label in enumerate(alpha):
        print('({}) Class {}:'.format(label_idx, label))
        for att in alpha[label]:
            print('\t{}: '.format(att), end=' ')
            for key in alpha[label][att].keys():
                print('{}={}'.format(key, alpha[label][att][key]), end=' ')
            print()
        print()

    # Quesion 5.1 (c)
    prior = {'<=50K': p_50k_lower, '>50K': p_50k_higher}
    result = test_samples(alpha, prior, test_line_num=10)
    print('Quesion 5.1 (c)')
    for line_idx in result:
        print('line-{}: '.format(line_idx), end='')
        line_result = result[line_idx]
        for label in line_result:
            prob = line_result[label]
            print('P({})={}'.format(label, prob), end=' ')
        print('The sample is classified to: {} \n'.format(
            get_key(line_result, max(line_result.values()))))
