import sys

__TRAIN_FILE = 'adult.data'
__TEST_FILE = 'adult.test'
__LABEL = ['<=50k', '>50k']

def train():
    train_sample = 0
    
    sample_50k_lower = .0
    sample_50k_higher = .0
    
    with open(__TRAIN_FILE) as train_file:
        # iterate over target frames
        for line_idx, line in enumerate(train_file.readlines()):
            values = line[:-1].split(', ')# omitting incomplete data lines
            if '?' in values:
                continue
            else:
                try:
                    age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, label = values
                except:
                    print(line_idx, values)
                else:
                    train_sample += 1
                    if label == '<=50K':
                        sample_50k_lower += 1
                    elif label == '>50K':
                        sample_50k_higher += 1
                    else:
                        print(label)
                        print('wrong training label')
                        sys.exit(0)
                
    p_50k_lower = sample_50k_lower / train_sample
    p_50k_higher = sample_50k_higher / train_sample

    print('Prior probability of each class:\n P(<=50)={}\n P(>50)={}\n'.format(p_50k_lower, p_50k_higher))

if __name__ == "__main__":
    train()