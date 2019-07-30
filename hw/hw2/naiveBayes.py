import sys

__TRAIN_FILE = 'adult.data'
__TEST_FILE = 'adult.test'
__ATTRIBUTE = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'label'
]
__CONTINUOUS_ATTRIBUTE = [
    'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
    'hours_per_week'
]


def get_samples():
    train_sample_num = 0

    sample = {
        att: [] if att in __CONTINUOUS_ATTRIBUTE else {}
        for att in __ATTRIBUTE
    }

    with open(__TRAIN_FILE) as train_file:
        # iterate over target frames
        for line_idx, line in enumerate(train_file.readlines()):
            values = line[:-1].split(', ')  # omitting incomplete data lines
            if '?' in values:
                continue
            else:
                age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, label = values

                # append continuous attributes
                sample['age'].append(int(age))
                sample['fnlwgt'].append(int(fnlwgt))
                sample['education_num'].append(int(education_num))
                sample['capital_gain'].append(int(capital_gain))
                sample['capital_loss'].append(int(capital_loss))
                sample['hours_per_week'].append(int(hours_per_week))

                # update discrete label
                if workclass in sample['workclass']:
                    sample['workclass'][workclass] += 1
                else:
                    sample['workclass'][workclass] = 1

                if education in sample['education']:
                    sample['education'][education] += 1
                else:
                    sample['education'][education] = 1

                if marital_status in sample['marital_status']:
                    sample['marital_status'][marital_status] += 1
                else:
                    sample['marital_status'][marital_status] = 1

                if occupation in sample['occupation']:
                    sample['occupation'][occupation] += 1
                else:
                    sample['occupation'][occupation] = 1

                if relationship in sample['relationship']:
                    sample['relationship'][relationship] += 1
                else:
                    sample['relationship'][relationship] = 1

                if race in sample['race']:
                    sample['race'][race] += 1
                else:
                    sample['race'][race] = 1

                if sex in sample['sex']:
                    sample['sex'][sex] += 1
                else:
                    sample['sex'][sex] = 1

                if native_country in sample['native_country']:
                    sample['native_country'][native_country] += 1
                else:
                    sample['native_country'][native_country] = 1

                if label in sample['label']:
                    sample['label'][label] += 1
                else:
                    sample['label'][label] = 1

                train_sample_num += 1

    return sample, train_sample_num


def get_prior(sample, sample_num):
    p_50k_lower = sample['label']['<=50K'] / sample_num
    p_50k_higher = sample['label']['>50K'] / sample_num

    print('Prior probability of each class:\n P(<=50)={}\n P(>50)={}\n'.format(
        p_50k_lower, p_50k_higher))

    return p_50k_lower, p_50k_higher


if __name__ == "__main__":
    sample, sample_num = get_samples()

    p_50k_lower, p_50k_higher = get_prior(sample, sample_num)